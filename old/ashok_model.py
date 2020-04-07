import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import numpy as np



converged = False
train3 = True
train12 = True
N = 80  #units per region
P = 4  #trial types
X = 2  #input classes
Y = 2  #output classes
B = 50  #batch size
Nepochs = 1000 #training epochs
Nepochspc = 50 #show progress every 50 epochs
lr = np.logspace(-3,-4,Nepochs) #learning rate
errthresh = 0.05 #break simulation once error reaches this level

px = 0.5 #connection probability across layers

dt = 0.02
tau = 0.1 #neuronal time constant
Tend = 4.
T = int(Tend/dt)
t = dt*np.arange(T)
s1start = 1.
s1end = 1.5

s2start = 3.
s2end = 3.5

sig0 = 0.05 #noise in initial state
sigeta = 0.05 #external noise on each timestep
sigetatest = 2*sigeta #noise during testing
alphaloss_activity = 0.0001 #penalty for activity

#external input (s1 is sample, s2 is test)
s1 = np.zeros([P,X])
s2 = np.zeros([P,X])
s1[0,:] = [1,0]
s1[1,:] = [0,1]
s1[2,:] = [1,0]
s1[3,:] = [0,1]
s2[0,:] = [1,0]
s2[1,:] = [0,1]
s2[2,:] = [0,1]
s2[3,:] = [1,0]

#output (left or right)
o = np.zeros([P,Y])
o[0,:] = [1,0]
o[1,:] = [1,0]
o[2,:] = [0,1]
o[3,:] = [0,1]

#rectified linear function
def relu(x):
    return x*(x>0)

#return inputs and outputs for one trial, given sample s1, test s2, and correct choice o
def genxy(s1,s2,o):
    X = len(s1)
    Y = len(o)


    goend = Tend


    x = np.zeros([T,X])
    y = np.zeros([T,Y])

    x[(t >= s1start) & (t < s1end)] = s1
    x[(t >= s2start) & (t < s2end)] = s2
    y[(t >= s2end) & (t < goend)] = o

    return x,y

#generate a batch of B trials
def gentrials(B,ustr=0,uinds=[]):
    x10 = sig0*relu(np.random.randn(B,N)).astype(np.float32)
    x20 = sig0*relu(np.random.randn(B,N)).astype(np.float32)
    x30 = sig0*relu(np.random.randn(B,N)).astype(np.float32)
    y0 = np.zeros([B,Y]).astype(np.float32)
    x = np.zeros([T,B,X]).astype(np.float32)
    u3 = np.zeros([T,B,N]).astype(np.float32)
    ytarg = np.zeros([T,B,Y]).astype(np.float32)

    u3[uinds,:,:] = ustr

    tt = np.zeros(B)

    for bi in range(B):
        ind = np.random.choice(P)
        tt[bi] = ind
        x[:,bi,:],ytarg[:,bi,:] = genxy(s1[ind,:],s2[ind,:],o[ind,:])

    return x10,x20,x30,y0,x,u3,ytarg,tt


#evaluate accuracy
def calcacc(sigetatest,ustr,uinds,output=True,Ntest=1000):
    x10,x20,x30,y0,x,u3,ytarg,tt = gentrials(Ntest,ustr,uinds)
    feed_dict = {x10_t: x10, x20_t: x20, x30_t: x30, y0_t: y0, x_t: x, u3_t: u3, ytarg_t: ytarg, sigeta_t: sigetatest}
    [x1,x2,x3,y],smy = sess.run([scan_out,smy_t],feed_dict=feed_dict)


    ybin = smy[T-1,:,:] > 0.9
    ybintarg = ytarg[T-1,:,:]
    err = ybin != ybintarg

    if output:
        print("error rate:",np.mean(err))
        print("\tAA:",np.mean(err[tt==0]))
        print("\tBB:",np.mean(err[tt==1]))
        print("\tAB:",np.mean(err[tt==2]))
        print("\tBA:",np.mean(err[tt==3]))
    return np.mean(err)

#maps state at time t to state at time t+1
def scan_step(prevstate,inputs):
    with tf.variable_scope('model',reuse=True):
        J1 = tf.get_variable('J1',shape=[N,N])
        J2 = tf.get_variable('J2',shape=[N,N])
        J3 = tf.get_variable('J3',shape=[N,N])
        J12 = tf.get_variable('J12',shape=[N,N])
        J12mask = tf.get_variable('J12mask',shape=[N,N])
        J23 = tf.get_variable('J23',shape=[N,N])
        J23mask = tf.get_variable('J23mask',shape=[N,N])
        wx = tf.get_variable('wx',shape=[X,N])
        wy = tf.get_variable('wy',shape=[N,Y])
        b1 = tf.get_variable('b1',shape=[N])
        b2 = tf.get_variable('b2',shape=[N])
        b3 = tf.get_variable('b3',shape=[N])

    x1prev,x2prev,x3prev,yprev = prevstate
    x,u3 = inputs
    B = tf.shape(x)[0]

    sigeta_t = tf.get_default_graph().get_tensor_by_name('sigeta:0')

    # x1 = (1.-dt/tau)*x1prev + (dt/tau)*tf.nn.relu(tf.matmul(x1prev,J1) + tf.matmul(x,wx) + b1 + sigeta_t*tf.random_normal([B,N]))
    # x2 = (1.-dt/tau)*x2prev + (dt/tau)*tf.nn.relu(tf.matmul(x2prev,J2) + tf.matmul(x1,J12*J12mask) + b2 + sigeta_t*tf.random_normal([B,N]))
    # x3 = (1.-dt/tau)*x3prev + (dt/tau)*tf.nn.relu(tf.matmul(x3prev,J3) + tf.matmul(x2,J23*J23mask) + b3 + sigeta_t*tf.random_normal([B,N]) + u3)
    # y = tf.matmul(x3,wy)

    # x = (1.-dt/tau)*x3prev + (dt/tau)*tf.nn.relu(tf.matmul(x3prev,J3) + tf.matmul(x2,J23*J23mask) + b3 + sigeta_t*tf.random_normal([B,N]) + u3)
    x1 = (1.-dt/tau)*x1prev + (dt/tau)*tf.nn.relu(tf.matmul(x1prev,J1) + tf.matmul(x,wx) + b1 + sigeta_t*tf.random_normal([B,N]))
    x2 = (1.-dt/tau)*x2prev + (dt/tau)*tf.nn.relu(tf.matmul(x2prev,J2) + tf.matmul(x1,J12*J12mask) + b2 + sigeta_t*tf.random_normal([B,N]))
    x3 = (1.-dt/tau)*x3prev + (dt/tau)*tf.nn.relu(tf.matmul(x3prev,J3) + tf.matmul(x2,J23*J23mask) + b3 + sigeta_t*tf.random_normal([B,N]) + u3)
    # x3 = x3prev
    y = tf.matmul(x3,wy)

    return [x1,x2,x3,y]

tf.reset_default_graph()

#initial condition
x10_t = tf.placeholder(dtype=tf.float32,shape=[None,N],name='x10')
x20_t = tf.placeholder(dtype=tf.float32,shape=[None,N],name='x20')
x30_t = tf.placeholder(dtype=tf.float32,shape=[None,N],name='x30')
y0_t = tf.placeholder(dtype=tf.float32,shape=[None,Y],name='y0')
#odor input
x_t = tf.placeholder(dtype=tf.float32,shape=[None,None,X],name='x')
#input to region 3 (ALM) that represents optogenetic inactivation
u3_t = tf.placeholder(dtype=tf.float32,shape=[None,None,N],name='u3')
#target
ytarg_t = tf.placeholder(dtype=tf.float32,shape=[None,None,Y],name='ytarg')

#learning rate
lr_t = tf.placeholder(dtype=tf.float32,shape=[],name='lr')
#magnitude of noise
sigeta_t = tf.placeholder(dtype=tf.float32,shape=[],name='sigeta')


#initial conditions for weights and biases
J10 = np.random.randn(N,N).astype(np.float32)/np.sqrt(N)
J20 = np.random.randn(N,N).astype(np.float32)/np.sqrt(N)
J30 = np.random.randn(N,N).astype(np.float32)/np.sqrt(N)
J120mask = (np.random.rand(N,N) < px).astype(np.float32)
J120 = (np.random.randn(N,N) * J120mask).astype(np.float32)/np.sqrt(px*N)
J230mask = (np.random.rand(N,N) < px).astype(np.float32)
J230 = (np.random.randn(N,N) * J230mask).astype(np.float32)/np.sqrt(px*N)
wx0 = np.random.randn(X,N).astype(np.float32)/np.sqrt(X)
wy0 = 2*np.random.randn(N,Y).astype(np.float32)/np.sqrt(N)
b10 = np.zeros(N,dtype=np.float32)
b20 = np.zeros(N,dtype=np.float32)
b30 = np.zeros(N,dtype=np.float32)

#variables ("mask" variables impose sparser connectivity between regions
with tf.variable_scope('model'):
	J1 = tf.get_variable('J1',initializer=J10,trainable=train12)
	J2 = tf.get_variable('J2',initializer=J20,trainable=train12)
	J3 = tf.get_variable('J3',initializer=J30,trainable=train3)
	J12 = tf.get_variable('J12',initializer=J120,trainable=False)
	J12mask = tf.get_variable('J12mask',initializer=J120mask,trainable=False)
	J23 = tf.get_variable('J23',initializer=J230,trainable=False)
	J23mask = tf.get_variable('J23mask',initializer=J230mask,trainable=False)
	wx = tf.get_variable('wx',initializer=wx0)
	wy = tf.get_variable('wy',initializer=wy0,trainable=False)
	b1 = tf.get_variable('b1',initializer=b10)
	b2 = tf.get_variable('b2',initializer=b20)
	b3 = tf.get_variable('b3',initializer=b30)

inputs_t = x_t,u3_t
initial_state_t = [x10_t,x20_t,x30_t,y0_t]
scan_out = tf.scan(scan_step,inputs_t,initial_state_t)
x1_t,x2_t,x3_t,y_t = scan_out
smy_t = tf.nn.softmax(y_t)

#loss is computed by comparing y and y's target during the go period
indgo = np.argwhere(t == s2end)[0][0]
ygo = y_t[(indgo+1):T,:,:]
ytarggo = ytarg_t[(indgo+1):T,:,:]
loss_err = tf.reduce_sum(tf.losses.softmax_cross_entropy(ytarggo,ygo))/B
#activity penalty
loss_activity = alphaloss_activity*tf.reduce_sum(tf.pow(x1_t,2) + tf.pow(x2_t,2) + tf.pow(x3_t,2))/(T*B*3*N)

train_step = tf.train.AdamOptimizer(lr_t).minimize(loss_err+loss_activity)

#initialize session (closes previous one if it exists)
try:
	sess
except NameError:
	sess = tf.InteractiveSession()
else:
	sess.close()
	sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#keep track of error over sessions
track_loss_err = np.zeros(Nepochs)

lastt = time.time()
count = 1
for ti in range(Nepochs):
    x10,x20,x30,y0,x,u3,ytarg,tt = gentrials(B)

    feed_dict = {x10_t: x10, x20_t: x20, x30_t: x30, y0_t: y0, x_t: x, u3_t: u3, ytarg_t: ytarg, lr_t: lr[ti], sigeta_t: sigeta}
    _,track_loss_err[ti] = sess.run([train_step,loss_err],feed_dict=feed_dict)
    if (ti % Nepochspc) == 0: #print percent complete, time since last print, and show plots
        curt = time.time()
        err1 = calcacc(sigetatest,0,[],output=False)
        print("\r" + str(int(100*ti/Nepochs)) + "%,", np.round(curt-lastt,2), "seconds, error:",err1, end="")
        [x1,x2,x3,y],smy = sess.run([scan_out,smy_t],feed_dict=feed_dict)

        lastt = curt
        plt.clf()
        plt.subplot(511)
        plt.semilogy(track_loss_err[0:ti])
        plt.xlim([0,Nepochs])
        plt.subplot(512)
        plt.plot(t,x1[:,0,:])
        plt.subplot(513)
        plt.plot(t,x2[:,0,:])
        plt.subplot(514)
        plt.plot(t,x3[:,0,:])
        plt.subplot(515)
        plt.plot(t,smy[:,0,0],"k")
        #plt.plot(t,ytarg[:,0,0],"gray")
        plt.plot(t,smy[:,0,1],"r")
        #plt.plot(t,ytarg[:,0,1],"pink")
        #plt.ylim(-.5,1.5)

        plt.pause(.0001)
        plt.show()
        count += 1
        if err1 < errthresh:
            converged = True
            break

#everything below is just visualization

def showtrial(tt1,sigetatest,ustr,uinds):
    x10,x20,x30,y0,x,u3,ytarg,tt = gentrials(100,ustr,uinds)
    feed_dict = {x10_t: x10, x20_t: x20, x30_t: x30, y0_t: y0, x_t: x, u3_t: u3, ytarg_t: ytarg, sigeta_t: sigetatest}
    x1,x2,x3,y = sess.run(scan_out,feed_dict=feed_dict)
    ti = np.argwhere(tt==tt1)[0][0]

    plt.subplot(411)
    plt.plot(t,x1[:,ti,:])
    plt.subplot(412)
    plt.plot(t,x2[:,ti,:])
    plt.subplot(413)
    plt.plot(t,x3[:,ti,:])
    plt.subplot(414)
    plt.plot(t,y[:,ti,0],"k")
    #plt.plot(t,ytarg[:,ti,0],"gray")
    plt.plot(t,y[:,0,1],"r")
    #plt.plot(t,ytarg[:,ti,1],"pink")
    #plt.ylim(-.5,1.5)

def showneuron(tt1,sigetatest,ustr,uinds,ind,c):
    x10,x20,x30,y0,x,u3,ytarg,tt = gentrials(100,ustr,uinds)
    feed_dict = {x10_t: x10, x20_t: x20, x30_t: x30, y0_t: y0, x_t: x, u3_t: u3, ytarg_t: ytarg, sigeta_t: sigetatest}
    x1,x2,x3,y = sess.run(scan_out,feed_dict=feed_dict)
    ti = np.argwhere(tt==tt1)[0][0]

    plt.subplot(131)
    plt.plot(t,x1[:,ti,ind],color=c)
    plt.subplot(132)
    plt.plot(t,x2[:,ti,ind],color=c)
    plt.subplot(133)
    plt.plot(t,x3[:,ti,ind],color=c)




sdinds = (t > s1start) & (t<s2start)
seinds = (t > s1start) & (t<(s1end+0.5))
ldinds = (t > (s1end+0.5)) & (t<s2start)

sigetatest = 0.2
ustr = -5.  # disruption for inhibition experiements
Ntest = 1000
print("control")
errc1 = calcacc(sigetatest,0,[],Ntest=Ntest)
print("sample+early delay")
errse1 = calcacc(sigetatest,ustr,seinds,Ntest=Ntest)
print("late delay")
errld1 = calcacc(sigetatest,ustr,ldinds,Ntest=Ntest)
print("sample+delay")
errsd1 = calcacc(sigetatest,ustr,sdinds,Ntest=Ntest)

plt.plot([1-errc1,1-errse1,1-errld1,1-errsd1],'.-')
plt.ylim(.5,1)
#plt.yticks([.85,.9,.95,1])
plt.xticks(np.arange(4),["Control","Sample+early","Late","Sample+delay"])
plt.ylabel("Accuracy")
plt.tight_layout()
#plt.savefig("plots/acc2.svg")

plt.figure()
showtrial(0,sigetatest,ustr,seinds)

plt.figure(figsize=cm2in(15,4))
ind = 22
ustr = 0
showneuron(0,sigetatest,ustr,sdinds,ind,"blue")
showneuron(1,sigetatest,ustr,sdinds,ind,"red")
showneuron(2,sigetatest,ustr,sdinds,ind,[.5,.5,1])
showneuron(3,sigetatest,ustr,sdinds,ind,[1,.5,.5])
plt.tight_layout()

#plt.savefig("plots/net1_neuron"+str(ind)+".svg")
plt.close("all")
