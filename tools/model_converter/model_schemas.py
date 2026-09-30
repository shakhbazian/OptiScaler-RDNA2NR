"""Pinned tensor schema, attributed to MLX-DLSS 0ca2deab (Apache-2.0)."""
import numpy as np

CHANNELS = {**dict.fromkeys(list(range(5))+list(range(66,71)),32),
            **dict.fromkeys(list(range(5,9))+list(range(62,66)),64),
            **dict.fromkeys(list(range(9,15))+list(range(56,62)),128),
            **dict.fromkeys(list(range(15,23))+list(range(48,56)),256)}
DOWN, UP = {4,8,14,22}, {48,56,62,66}


def schemas():
    """Each tuple is role, encoding, storage shape, address interpretation."""
    result = {}
    def matrix(role,k,n,layout='qmma-k'):
        return (role,'e4m3fn',[k,n],layout)
    def vector(role,n,encoding='fp16'):
        return (role,encoding,[n],'linear')
    def zero(n=16):
        return ('alignment','zero',[n],'linear')
    def size(parts):
        return sum(int(np.prod(shape))*{'e4m3fn':1,'zero':1,'fp16':2,'fp32':4}[dtype]
                   for _,dtype,shape,_ in parts)
    def put(block,layer,parts):
        result[f'block{block}.layer{layer}.layer'] = parts
    for b,c in CHANNELS.items():
        p = ([matrix('ffn_combined',c,4*c+128,'qmma-n128'),matrix('ffn_output',c,c)]
             if c>=64 else [matrix('ffn_expand',c,128),matrix('ffn_contract',128,c)])
        if b in UP:
            p += [matrix('upsample_projection',2*c,c)]
        if b not in UP or c==32:
            p += [zero()]
        if b==0:
            p += [('input_adapter','fp16',[16,c],'hmma'),vector('ffn_cos_skip',c),zero()]
        else:
            p += [vector('ffn_cos_skip',c)]
        if b in UP:
            if c==32:
                p += [zero()]
            p += [vector('upsample_sin',c)]
        elif b==70:
            p += [vector('input_merge_sin',c),vector('input_merge_cos',c)]
        elif b!=0:
            p += [zero()]
        p += [matrix('qkv_pre_semantic',c,3*c),
              ('attention_bias','fp16',[c//32,64,64],'window-bias'),
              vector('attention_scale',c//32,'fp32')]
        padding = (-size(p))%16
        if padding:
            p += [zero(padding)]
        p += [matrix('attention_projection',c,c),vector('attention_cos_skip',c)]
        if b in DOWN:
            p += [matrix('downsample_projection',c,2*c)]
            expected = {32:0x58c0,64:0x11130,128:0x38230,256:0xc8440}[c]
            if expected>size(p):
                p += [zero(expected-size(p))]
        elif b==70:
            p += [zero(),('out_gain','fp16',[16,16],'hmma-output4'),
                  ('out_conv','fp16',[16,16],'hmma-output4')]
        else:
            p += [zero()]
        put(b,0,p)
    for b in list(range(23,31))+list(range(40,48)):
        p = [matrix('first_projection',512,512)]
        p += [matrix(f'group_expand_{i}',64,256) for i in range(8)]
        p += [matrix(f'group_project_{i}',256,64) for i in range(8)]
        put(b,0,p)
        put(b,1,[matrix('weight3',512,512),vector('ffn_cos_skip',512)])
        put(b,2,[matrix('qkv_pre_semantic',512,1536),
                 ('attention_bias','fp16',[16,64,64],'window-bias'),
                 vector('attention_scale',16,'fp32')])
        put(b,3,[matrix('attention_projection',512,512),vector('attention_cos_skip',512)])
    put(30,4,[matrix('global_input_projection',512,1024),zero()])
    for b in range(31,39):
        put(b,0,[matrix('ffn_expand',1024,4096),zero()])
        put(b,1,[matrix('ffn_contract',4096,1024),vector('ffn_cos_skip',1024)])
        put(b,2,[vector('attention_scale',32,'fp32'),matrix('qkv_pre_semantic',1024,3072)])
        put(b,3,[vector('attention_scalar_unresolved',1)])
        put(b,4,[matrix('attention_projection',1024,1024),vector('attention_cos_skip',1024)])
    put(39,0,[matrix('bridge_projection',1024,512),vector('bridge_sin',512)])
    result['block70.layer0.blend_scale'] = [vector('blend_scale_unresolved',1)]
    return result
