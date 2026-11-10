from max.algorithm import parallelize

comptime U64Ptr = Pointer[UInt64, AnyOrigin[mut=True]]


def main() raises:
    var n = 8
    def worker(i: Int) {imm}:
        print(i)
    parallelize(worker, n, 2)
