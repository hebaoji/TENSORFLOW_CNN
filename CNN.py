import numpy as np
import scipy
import matplotlib
import tensorflow as tf
#####################CONTACT MAP, SOLVENT ACC, SECONDARY STRUCTURE and COLSTAT###################
def CNN2d(x,W,b,stride=1):
	x=tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')
	return tf.nn.relu(x+b)

def maxpool2d(x,k=2):
	return tf.nn.max_pool(x,ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def maxpool2dcol(x,k=2,m=21):
	return tf.nn.max_pool(x,ksize=[1,k,m,1],strides=[1,k,k,1],padding='SAME')
def maxpool(x,k,l,m,n):
	return tf.nn.max_pool(x,ksize=[1,k,l,1],strides=[1,m,n,1],padding='SAME')
##################################################################################################




if __name__ =='__main__':
##################################WEIGHT FACTOR########################################
##############################################contact map in two dimension
	weights={'contactmaplayer1':tf.Variable(tf.random_normal([5,5,11,16])),
		'contactmaplayer2':tf.Variable(tf.random_normal([3,3,16,32])),
#############################################SECONDARY structure in 2 D
		'SS1':tf.Variable(tf.random_normal([5,3,2,6])),
		'SS2':tf.Variable(tf.random_normal([3,3,6,18])),
#############################################solvent accessibilitty in 2D
		'SA1':tf.Variable(tf.random_normal([5,1,2,4])),
		'SA2':tf.Variable(tf.random_normal([3,1,4,8])),
##############################################colstats in 2D
		'COL1':tf.Variable(tf.random_normal([5,5,2,5])),
		'COL2':tf.Variable(tf.random_normal([3,3,5,10])),
#############################
		'SF1':tf.Variable(tf.random_normal([504,100])),
		'SF2':tf.Variable(tf.random_normal([100,2]))}
	biases={'contactmapbiase1':tf.Variable(tf.random_normal([16])),
		'contactmapbiase2':tf.Variable(tf.random_normal([32])),
		'SSbiase1':tf.Variable(tf.random_normal([6])),
		'SSbiase2':tf.Variable(tf.random_normal([18])),
		'SAbiase1':tf.Variable(tf.random_normal([4])),
		'SAbiase2':tf.Variable(tf.random_normal([8])),
		'COLbiase1':tf.Variable(tf.random_normal([5])),
		'COLbiase2':tf.Variable(tf.random_normal([10])),
		'SFbiase1':tf.Variable(tf.random_normal([100])),
		'SFbiase2':tf.Variable(tf.random_normal([2]))}

##############################PLACE HOLDER#############################################
	contactm=tf.placeholder("float",shape=[None,11,11,11])
	native=tf.placeholder("float",[None,2])
	SS=tf.placeholder("float",[None,11,3,2])
	SA=tf.placeholder("float",[None,11,1,2])
	colstats=tf.placeholder("float",[None,11,21,2])
	referen=tf.placeholder("float",[None,1])
	keep_prob = tf.placeholder("float")
###########################################################################

####################flowchart(1)
	conlayer1=CNN2d(contactm,weights['contactmaplayer1'],biases['contactmapbiase1'],1)
	conmaxpool1=maxpool2d(conlayer1,2)
	conlayer2=CNN2d(conmaxpool1,weights['contactmaplayer2'],biases['contactmapbiase2'],1)
	conmaxpool2=maxpool2d(conlayer2,2)
####################flowchart(2)
	SSlayer1=CNN2d(SS,weights['SS1'],biases['SSbiase1'],1)
	SSmaxpool1=maxpool(SSlayer1,2,1,2,1)
	SSlayer2=CNN2d(SSmaxpool1,weights['SS2'],biases['SSbiase2'],1)
	SSmaxpool2=maxpool(SSlayer2,2,1,2,1)
####################flowchart(3)
	SAlayer1=CNN2d(SA,weights['SA1'],biases['SAbiase1'],1)
	SAmaxpool1=maxpool(SAlayer1,2,1,2,1)
	SAlayer2=CNN2d(SAmaxpool1,weights['SA2'],biases['SAbiase2'],1)
	SAmaxpool2=maxpool(SAlayer2,2,1,2,1)	
###################flowchart(4)
	collayer1=CNN2d(colstats,weights['COL1'],biases['COLbiase1'],1)
	colmaxpool1=maxpool(collayer1,2,7,2,7)
	collayer2=CNN2d(colmaxpool1,weights['COL2'],biases['COLbiase2'],1)
	colmaxpool2=maxpool(collayer2,2,3,2,3)
##################################################RESHAPE
	
	PART1=tf.reshape(conmaxpool2,[-1,3*3*32])#3*3*32
	PART2=tf.reshape(SSmaxpool2,[-1,3*3*18])#3*3*18
	PART3=tf.reshape(SAmaxpool2,[-1,3*1*8])#3*1*8
	PART4=tf.reshape(colmaxpool2,[-1,3*10])#3*10
	SFIN=tf.concat(1,[PART1,PART2,PART3,PART4])
	PARTSF1=tf.nn.relu(tf.matmul(SFIN,weights['SF1'])+biases['SFbiase1'])
	PARTSF1drop=tf.nn.dropout(PARTSF1, keep_prob)
	PARTSF2=tf.nn.softmax(tf.matmul(PARTSF1drop, weights['SF2']) + biases['SFbiase2'])
######################################################################GRAPHE
	cross_entropy = -tf.reduce_sum(native * tf.log(PARTSF2))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_predict1 = tf.equal(tf.argmax(native, 1),tf.argmax(PARTSF2,1))
	correct_predict=tf.mul(referen,tf.cast(correct_predict1, "float"))
	accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
##################################################

		for i in range(2000):
			rang='SHORT'
#############CONSHAPE[1000,11,11,11]
			LI=np.random.randint(0,18)
			name=rang+str(LI)+'.npy'
			tmp=np.load(name)
			CONBATCH=np.zeros((10000,11,11,11))
			for j in range(10000):
				CONBATCH[j,:,:,:]=tmp[:,:,:,j]
#############SS SHAPE[1000,11,3,2]
			name=rang+str(LI)+'-SS.npy'
			tmp=np.load(name)
			SSBATCH=np.zeros((10000,11,3,2))
			for j in range(10000):
				SSBATCH[j,:,:,0]=tmp[j,0:11,:]
				SSBATCH[j,:,:,1]=tmp[j,11:,:]
##############SA SHAPE[[10000,11,2]]
			name=rang+str(LI)+'-SOLV.npy'
			tmp=np.load(name)
			SABATCH=np.zeros((10000,11,1,2))
			for j in range(10000):
				SABATCH[j,:,0,0]=tmp[j,0:11]
				SABATCH[j,:,0,1]=tmp[j,11:]
################COLS SHAPE[10000,11,21,2]
			name=rang+str(LI)+'-COLSTATS.npy'
			tmp=np.load(name)
			COLBATCH=np.zeros((10000,11,21,2))
			for j in range(10000):
				COLBATCH[j,:,:,0]=tmp[j,0:11,:]
				COLBATCH[j,:,:,1]=tmp[j,11:,:]
##################native SHAPE[10000,2]
			name=rang+str(LI)+'-Truelabel.npy'
			tmp=np.load(name)
			NATIVE1=np.zeros((10000,1))
			NATIVEBATCH=np.zeros((10000,2))
			count=0
			for j in range(10000):
				if(tmp[j,0]==0 and tmp[j,1]==0):
					NATIVEBATCH[j,0]=0.384
					NATIVEBATCH[j,1]=0
					NATIVE1[j,0]=1
					count+=1
				else:
					NATIVEBATCH[j,0]=0
					NATIVEBATCH[j,1]=0.616
					NATIVE1[j,0]=1
			for j in range(10000):
				if(tmp[j,0]==0 and tmp[j,1]==0):		
					NATIVE1[j,0]=(10000.0/(2*count))
#					print 10000.0/(2*count)
				else:
					NATIVE1[j,0]+=((2*count-10000.0)/(10000.0-count))
					NATIVE1[j,0]*=(10000.0/(2*count))
			tmp=np.zeros(1)
			print NATIVE1[:,0].sum()
#			print CONBATCH.shape, SSBATCH.shape,SABATCH.shape,COLBATCH.shape,NATIVEBATCH.shape			
###################################################################)
			if i%1 == 0:
				print count,10000-count
				train_accuracy = accuracy.eval(feed_dict={contactm:CONBATCH, SS:SSBATCH, SA:SABATCH, colstats:COLBATCH, native:NATIVEBATCH, keep_prob:1.0,referen:NATIVE1})
				tmp=sess.run(correct_predict1,feed_dict={contactm:CONBATCH, SS:SSBATCH, SA:SABATCH, colstats:COLBATCH, native:NATIVEBATCH, keep_prob:1.0,referen:NATIVE1})
				tmp1=np.array(tmp)
				print tmp1.sum()/10000.0
				print "step %d, training accuracy %g" % (i, train_accuracy)
			train_step.run(feed_dict={contactm:CONBATCH, SS:SSBATCH, SA:SABATCH, colstats:COLBATCH, native:NATIVEBATCH, keep_prob:1.0})	
		#	print "test accuracy %g" % accuracy.eval(feed_dict={contactm:CONBATCH, SS:SSBATCH, SA:SABATCH, colstats:COLBATCH, native:NATIVEBATCH, keep_prob:1.0})



