from benchmarks.siesta_benchmark import make_cl_benchmark


def main():
    myBenchmark = make_cl_benchmark()

    print(myBenchmark.classes_in_experience)
    print(myBenchmark.n_experiences)


if __name__ == "__main__":
    main()

