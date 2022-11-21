import cinn
import numpy as np
import time
from cinn import runtime

# from PIL import Image
 
# def getimg(fname):
    
#     image=Image.open(fname)
#     image= np.array(image)
#     return image

# print(getimg("/root/RemoteWorking/huazhibin.webp"))


def randomImgNCHW(w,h,c=3,n=1):
    return np.random.randint(0,255,(n,c,h,w))


def runCinn(C,args,n="matmul"):
    stages = cinn.create_stages([C])

    target = cinn.Target()
    builder = cinn.Module.Builder(n, target)

    func = cinn.lower(n, stages, [A.to_tensor(), B.to_tensor(), C])

    print(func)

    builder.add_function(func)
    module = builder.build()

    jit = cinn.ExecutionEngine()
    jit.link(module)

    exefn = jit.lookup(n)
    exefn(args)

m = cinn.Expr(4)
n = cinn.Expr(3)
k = cinn.Expr(2)


A = cinn.Placeholder("float32", "A", [m, k])
B = cinn.Placeholder("float32", "B", [k, n])

C = cinn.compute([
    m, n
], lambda v: A(v[0],v[1]) + B(v[0],v[1]),  "C")


a = runtime.cinn_buffer_t(

    np.arange(1,m.int()*k.int()+1).reshape(m.int(), k.int()).astype("float32"),

    runtime.cinn_x86_device)

b = runtime.cinn_buffer_t(
    np.arange(1,k.int()*n.int()+1).reshape(k.int(), n.int()).astype("float32"),
    runtime.cinn_x86_device)

    
c = runtime.cinn_buffer_t(
    np.zeros([m.int(), n.int()]).astype("float32"), runtime.cinn_x86_device)

args = [runtime.cinn_pod_value_t(_) for _ in [a, b, c]]


runCinn(C,args)

npa = a.numpy()
npb = b.numpy()
npc = c.numpy().astype("int32")

npr =np.random.randint(0,255,(6,8,3))

# print("a:",npa,npa.shape)
# print("b:",npb,npb.shape)
# print("c:",npc,npc.shape)

# print("npr:",npr)

t = randomImgNCHW(8,6)

print("npr-transpose:",t)

