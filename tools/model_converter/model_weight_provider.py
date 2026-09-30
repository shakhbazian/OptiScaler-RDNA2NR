"""Own bounded logical-weight provider for the pinned 71-block candidate graph."""
import numpy as np
from model_schemas import CHANNELS,schemas
from qmma_layout import decode_e4m3,unpack
from research_layouts import bias_offsets,hmma_offsets,n128_offsets
from weights_ht import load_file


def semantic_qkv(interleaved, heads):
    channels, columns = interleaved.shape
    if channels % heads or columns != channels * 3:
        raise ValueError('QKV shape/head mismatch')
    width = channels // heads
    order = np.array([head * 3 * width + kind * width + channel
                      for kind in range(3) for head in range(heads)
                      for channel in range(width)])
    return np.ascontiguousarray(interleaved[:, order])


class ModelWeightProvider:
    def __init__(self,source_path,expected_sha256):
        self.selection,self.data,_,records=load_file(source_path,expected_sha256)
        self.records={r['name']:r for r in records};self.contract=schemas()

    def record(self,name):
        record=self.records[name];start=record['sourceOffset'];payload=memoryview(self.data)[start:start+record['byteLength']]
        result={};offset=0
        for role,dtype,shape,layout in self.contract[name]:
            count=int(np.prod(shape));unit={'e4m3fn':1,'zero':1,'fp16':2,'fp32':4}[dtype]
            raw=payload[offset:offset+count*unit];offset+=count*unit
            if dtype=='zero':
                if any(raw):raise ValueError(f'nonzero padding in {name}')
                continue
            values=np.frombuffer(raw,{'e4m3fn':'u1','fp16':'<f2','fp32':'<f4'}[dtype])
            if layout=='qmma-k':values=unpack(raw,*shape)
            elif layout=='qmma-n128':values=values[n128_offsets(*shape)]
            elif layout=='window-bias':values=values.reshape(shape[0],4096)[:,bias_offsets()]
            elif layout.startswith('hmma'):
                values=values[hmma_offsets(shape[1])]
                if layout=='hmma-output4':values=values[:,:4]
            else:values=values.reshape(shape)
            if dtype=='e4m3fn':values=decode_e4m3(values).astype('<f2')
            result[role]=np.ascontiguousarray(values)
        if offset!=len(payload):raise ValueError(f'unconsumed record {name}')
        return result

    def __call__(self,block):
        if block in CHANNELS:
            c=CHANNELS[block];s=self.record(f'block{block}.layer0.layer');heads=c//32
            result={'family':'simple32' if c==32 else 'branched','ffn_cos':s['ffn_cos_skip'],
                    'qkv':semantic_qkv(s['qkv_pre_semantic'],heads),'bias':s['attention_bias'],
                    'scale':s['attention_scale'].astype('<f2'),'attention_projection':s['attention_projection'],
                    'attention_cos':s['attention_cos_skip']}
            if c==32:result.update(ffn_expand=s['ffn_expand'],ffn_contract=s['ffn_contract'])
            else:
                g=c//32;grouped=s['ffn_combined'].reshape(c,4*g+4,32)
                result['ffn_expand']=np.stack([[[grouped[i*32:(i+1)*32,o*4+b] for i in range(g)] for b in range(4)] for o in range(g)])
                result['ffn_branch']=np.stack([[grouped[o*32:(o+1)*32,4*g+b] for b in range(4)] for o in range(g)])
                result['ffn_output']=s['ffn_output']
            for source,target in (('input_adapter','adapter'),('downsample_projection','downsample'),
                                  ('upsample_projection','upsample'),('upsample_sin','upsample_sin'),
                                  ('input_merge_sin','merge_sin'),('input_merge_cos','merge_cos'),
                                  ('out_gain','out_gain'),('out_conv','out_conv')):
                if source in s:result[target]=s[source]
            return result
        if 23<=block<=30 or 40<=block<=47:
            s0=self.record(f'block{block}.layer0.layer');s1=self.record(f'block{block}.layer1.layer')
            s2=self.record(f'block{block}.layer2.layer');s3=self.record(f'block{block}.layer3.layer')
            result={'family':'split','ffn_first':s0['first_projection'],
                    'ffn_expand':np.stack([s0[f'group_expand_{i}'] for i in range(8)]),
                    'ffn_group_project':np.stack([s0[f'group_project_{i}'] for i in range(8)]),
                    'ffn_output':s1['weight3'],'ffn_cos':s1['ffn_cos_skip'],
                    'qkv':semantic_qkv(s2['qkv_pre_semantic'],16),'bias':s2['attention_bias'],
                    'scale':s2['attention_scale'].astype('<f2'),'attention_projection':s3['attention_projection'],
                    'attention_cos':s3['attention_cos_skip']}
            if block==30:result['global_projection']=self.record('block30.layer4.layer')['global_input_projection']
            return result
        if 31<=block<=38:
            s0=self.record(f'block{block}.layer0.layer');s1=self.record(f'block{block}.layer1.layer')
            s2=self.record(f'block{block}.layer2.layer');s4=self.record(f'block{block}.layer4.layer')
            return {'family':'global','ffn_expand':s0['ffn_expand'],'ffn_contract':s1['ffn_contract'],
                    'ffn_cos':s1['ffn_cos_skip'],'qkv':semantic_qkv(s2['qkv_pre_semantic'],32),
                    'scale_fp32':s2['attention_scale'],'attention_projection':s4['attention_projection'],
                    'attention_cos':s4['attention_cos_skip']}
        if block==39:
            s=self.record('block39.layer0.layer');return {'family':'bridge','bridge_projection':s['bridge_projection'],'bridge_sin':s['bridge_sin']}
        raise ValueError(f'unsupported block {block}')
