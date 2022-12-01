import subprocess
import re
import matplotlib
import matplotlib.pyplot as plt

SOURCE_FILE = "matrixMultiple.cu"
EXE_FILE = "matrix-mut"

def replace_define(filename, block_size):
    content = None
    with open(filename) as f:
        content = f.read()
    
    content = re.sub(r'#define BLOCK_SIZE \d+', "#define BLOCK_SIZE {}".format(block_size), content)

    new_filename = "rep_{}".format(filename)

    with open(new_filename, "w") as f:
        f.write(content)

    return new_filename

def plot():
    block_size_list = []
    t2_list = []
    t3_list = []
    for block_size in range(1,33,1):
        block_size_list.append(block_size)
        new_source = replace_define(SOURCE_FILE, block_size)
        compile_cmd = "nvcc -lcudart -lcublas {} -o {}".format(
            new_source, EXE_FILE
        )
        subprocess.check_call(compile_cmd, shell=True)
        output = subprocess.check_output("./{}".format(EXE_FILE), shell=True)
        print(block_size, ": ", output)
        t1s, t2s, t3s, t4s = output.decode().strip().split(",")

        t2_list.append(float(t2s.strip()))
        t3_list.append(float(t3s.strip()))
    
    plt.plot(block_size_list, t2_list, ".-", label="normal")
    plt.plot(block_size_list, t3_list, ".-", label="tiled")

    plt.xlabel("Block Size")
    plt.ylabel("Calulation Time (s)")
    plt.legend()
    plt.show()

def block_size():
    for i in range(1,33,1):
        bs = i**2
        if bs < 96:
            continue
        if bs % 32 != 0:
            continue
        if 1536 % bs != 0:
            continue
        print(i)


if __name__ == "__main__":
    block_size()