using System;
using System.IO;

public static class SyntheticFrame
{
    public static void Write(string path, int width, int height)
    {
        using (var output = File.Create(path))
        {
            var row = new byte[width * 8];
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    int offset = x * 8;
                    ushort r = (ushort)(0x3400 + ((x * 13 + y * 7) % 24) * 64);
                    ushort g = (ushort)(0x3600 + ((x * 5 + y * 17) % 24) * 64);
                    ushort b = (ushort)(0x3800 + ((x * 11 + y * 3) % 16) * 64);
                    row[offset] = (byte)r;
                    row[offset + 1] = (byte)(r >> 8);
                    row[offset + 2] = (byte)g;
                    row[offset + 3] = (byte)(g >> 8);
                    row[offset + 4] = (byte)b;
                    row[offset + 5] = (byte)(b >> 8);
                    row[offset + 6] = 0x00;
                    row[offset + 7] = 0x3c;
                }
                output.Write(row, 0, row.Length);
            }
        }
    }
}
