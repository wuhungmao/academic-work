# Generating custom dataset

Although you are not expected to create custom datasets, it may be useful to
test your code with a smaller dataset where the output is known. In order
to assist those who want to create custom datasets for testing, here is the
dataset layout that we use:

```
student count
ta count
student structs
ta structs
```

Note that the values are binary and newlines are not part of the file layout.
As such, you may need to write custom program to create said datasets.
So, if you were to have 10 students with 30 ta contracts, then the size of the
file will be:

`sizeof(int) * 2 + sizeof(student struct) * 10 + sizeof(ta struct) * 30`

