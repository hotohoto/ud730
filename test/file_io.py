filename = "hello.txt"

with open(filename, "w") as f:
    f.write("Hello World")

with open(filename) as f:
    print(f.read())
