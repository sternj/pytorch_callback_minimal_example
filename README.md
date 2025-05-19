# Example of unexpected thread-local callback behavior

When removing a thread-local callback with `RecordScope::BACKWARD_FUNCTION` from the autograd thread, subsequent tasks' thread-local state blocks squash the removal. This seems to be by design, as the thread-local state is [squashed in the `ThreadLocalStateGuard` destructor](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/ThreadLocalState.h#L106) and [reset with each new graph task](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/engine.cpp#L558). How is this meant to be used, and is there a canonical way of removing a callback in the middle of the backward pass?



Run the example with `python -m cb_test`-- [this assertion error](./src/cpp/cb_test/libcbtest.cpp#L36) is hit when I wouldn't expect it to be. 


## Dependencies

CMake, the CUDA toolkit, and Ninja all need to be moderately recent versions-- if it builds the error should occur, but I made this with:
- Python 3.12.10
- clang 18.1.8
- cmake 3.31.7
- cuda 12.8
- ninja 1.12.1