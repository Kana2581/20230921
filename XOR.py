import numpy as np
sgn=lambda x :np.array(x>0,dtype=np.int32)
createArr=lambda x,y:np.array([[x,y]])
def twoLayerPerceptron(arr):
    w1=np.array([[-1,1],[-1,1]])
    b1=np.array([[1.5,-0.5]])
    w2=np.array([[1],[1]])
    b2=np.array([[-1.5]])
    return sgn(np.dot(sgn(np.dot(arr,w1)+b1),w2)+b2)
matrix=np.array([[0,0],[0,1],[1,0],[1,1]])
print(twoLayerPerceptron(matrix))

