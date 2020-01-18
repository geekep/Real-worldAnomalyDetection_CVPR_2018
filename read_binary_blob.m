function [s, data] = read_binary_blob(fn)

f = fopen(fn, 'r');
% s: size of the blob, e.g. num * chanel * length * height * width
s = fread(f, [1 5], 'int32');
m = s(1) * s(2) * s(3) * s(4) * s(5);
% data: blob binary data in single precision, e.g. float in C++
data = fread(f, [1 m], 'single');
fclose(f);

end

