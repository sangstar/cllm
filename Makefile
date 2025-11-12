
CXX = gcc

build: main.c deserialize.c
	$(CXX) $^ -o cllm