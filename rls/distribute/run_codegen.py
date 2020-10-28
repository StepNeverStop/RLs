import glob
from grpc_tools import protoc

for proto in glob.glob(r'./protos/*.proto'):
    print(proto)
    protoc.main((
        '',
        r'-I./protos',
        r'--python_out=./pb2',
        r'--grpc_python_out=./pb2',
        proto,
    ))