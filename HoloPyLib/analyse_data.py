import numpy as np
import numba
import pandas as pd
import matplotlib.pyplot as plt
import os


from scipy.optimize import curve_fit
from numpy import linalg as la
from time import sleep
from time import time
from progress.bar import Bar



#tagada


#####################################################################
############# 									        ############# 
#############  créer les dossiers de stockage des datas #############   
############# 									        ############# 
#####################################################################


path ='./'
data_folder_name="data_trajectories/"
data_folder = path + data_folder_name

volume_folder="volume_trajectories/"
speed_folder="speed_trajectories/"
angle_folder="angle_trajectories/"
MSD_folder="MSD_trajectories/"
MSD_txt_folder="MSD_trajectories/MSD_txt/"
filter_MSD_txt="MSD_trajectories/MSD_filter_txt/"
traj_folder="trajectories/"

List_folder=[volume_folder,speed_folder, angle_folder,MSD_folder,MSD_txt_folder,filter_MSD_txt,traj_folder]


def create_dir():
	
	if not os.path.exists(data_folder):
		os.makedirs(data_folder)
		
		for folder_name in List_folder:
			direction=data_folder+folder_name
			os.makedirs(direction)
			
	return







##############################################################################
############# 									                 ############# 
#############   METHODE DE FILTRAGE AUTOMATIQUE DES TRAJECTOIRES #############      
############# 									                 ############# 
##############################################################################




############# récupérer la trajectoire unique ############# 

def get_xyz(file_traj,index):
	l=len(file_traj)
	lone_traj=[]
	
	for i in range(l):
		if file_traj[i][5]==index:
			lone_traj.append(file_traj[i])
	lone_traj=np.array(lone_traj,dtype=np.float32)	
	
	T=lone_traj[:,0] #time
	Z=lone_traj[:,1] 
	Y=lone_traj[:,2]
	X=lone_traj[:,3]
	Pix=lone_traj[:,4] #nbVoxel
	
	return X,Y,Z,Pix,T

	
	
	
############# récupérer l'ensemble des numéros de trajectoires ############# 	
	
def get_all_index(file_traj):
	l=len(file_traj)
	liste_all_index=[]
	
	liste_all_index.append(file_traj[0][5])
	
	for i in range(1,l):
		if file_traj[i][5] not in liste_all_index :
			liste_all_index.append(file_traj[i][5])
	
	return liste_all_index


############# détecter les corrélation entre x,y et z pour le filtrage des trajectoires ############# 

def correlation_xyz(file_traj,index):
	xyz=get_xyz(file_traj,index)
	
	x=(xyz[0]-np.min(xyz[0]))/(np.max(xyz[0])-np.min(xyz[0]))
	y=(xyz[1]-np.min(xyz[1]))/(np.max(xyz[1])-np.min(xyz[1]))
	z=(xyz[2]-np.min(xyz[2]))/(np.max(xyz[2])-np.min(xyz[2]))

	l=len(xyz[0])
	data = {'x': x,'y': y,'z': z}
	dataframe_speed = pd.DataFrame(data, columns=['x', 'y', 'z'])
	matrix = dataframe_speed.corr()
	
	return matrix,l,


	




 
############# filtrage sélectif sur la longueur et la correlation de la trajectoire  #############  

def filtrage_method_1(file_traj,seuil_correl=2.9,seuil_correl_inf=2.3,dmin=20, d_inf=5):
	
	particles_index = file_traj['particle'].unique().tolist()
	particles_interest=[]
	
	bar = Bar('Loading', max=len(particles_index))
	
	for index in particles_index:
		t0=time()
		
		xyz=np.array(get_xyz(file_traj,index))
		Tmax=len(xyz[0])
	
		step=2
		
		x=xyz[0]
		y=xyz[1]
		z=xyz[2]
		
		Ltot2=((x[step:]-x[:-step])**2+(y[step:]-y[:-step])**2+(z[step:]-z[:-step])**2).sum()
		L_tot_real=np.sqrt(Ltot2)
		
		corr_data=correlation_xyz(file_traj,index)
		correlation_traj=np.array(corr_data[0])
		norm = la.norm(correlation_traj) 
		
		# seuil sur la norme et sur la distance parcourue
		if norm>seuil_correl and L_tot_real>d_inf:
			particles_interest.append(int(index))
		
		# seuil sur la distance maximale parcourue
		if L_tot_real>dmin and norm>seuil_correl_inf :
			particles_interest.append(index)
		t1=time()
		sleep(t1-t0)
		bar.next()
	bar.finish()
		
	return particles_interest




############# calcul de la distance maximale et de la diffusion brownienne  #############  	


def dist_max_Ldiffusion(file_traj,index,dt=0.01):
	
	Kb=1.380649*10**(-23)
	T=273+20; r=1*10**(-6); mu=1*10**(-3)
	D=Kb*T/(6*np.pi*r*mu)
	
	
	
	xyz=np.array(get_xyz(file_traj,index),dtype=np.float32)
	x=xyz[0]
	y=xyz[1]
	z=xyz[2]
	Tmax=len(x)
	list_dist=[]
	Matrix_ones=np.ones(Tmax)
	list_dist=np.sqrt((x-Matrix_ones*x[0])**2+(y-Matrix_ones*y[0])**2+(z-Matrix_ones*z[0])**2)
	dist_max=list_dist.max()
	
	delta_t=Tmax*dt
	L_carac=np.sqrt(6*delta_t*D)*10**6
	
	return dist_max, L_carac
	
	

############# filtrage avec critère sur la distance maximale au point initial et la diffusion  #############  	
	
		
def filtrage_method_2(file_traj):
	
	particles_index=[]
	particles_interest=[]
	
	for i in range(len(file_traj)):
		
		if file_traj[i][5] not in particles_index:
			particles_index.append(file_traj[i][5])
	
	bar = Bar('Processing', max=len(particles_index))
	
	for index in particles_index:
		t0=time()
	
		data_filter = dist_max_Ldiffusion(file_traj,index)
		dist_max=data_filter[0]
		L_carac=data_filter[1]


		if dist_max>20*L_carac:
			particles_interest.append(index)
		
		t1=time()
		sleep(t1-t0)
		bar.next()
	bar.finish()
		
	return particles_interest
	
			
		
############# filtrage final par concaténation des 2 méthodes #############  
		
def get_good_index(file_traj, seuil_correl=2.6):

	#sinon pas fichier
	print('----------')
	print('filtering method 1')
	particles_filtered_1=filtrage_method_1(file_traj)
	print('\n'+'Result 1 =')
	print(particles_filtered_1)
	print('\n')
	
	
	print('----------')
	print('filtering method 2')
	particles_filtered_2=filtrage_method_2(file_traj)
	print('\n'+'Result 2 =')
	print(particles_filtered_2)
	print('\n')
	
	difference_particules=[]
	
	for index_new in particles_filtered_2:
		if index_new not in particles_filtered_1:
			difference_particules.append(index_new)
		
	bar = Bar('Loading trajectories of interest', max=len(difference_particules))
	for traj_i in difference_particules:
		t0=time()
		corr_data=correlation_xyz(file_traj,traj_i)
		correlation_traj=np.array(corr_data[0],dtype=np.float32)
		norm = la.norm(correlation_traj) 
		
		# seuil sur la norme
		if norm>seuil_correl :
			particles_filtered_1.append(traj_i)
		t1=time()
		sleep(t1-t0)
		bar.next()
	bar.finish()
	print('\n')
	particles_filtered_1.sort()
	
	
	path_fig=path + data_folder_name
	txt_name='particles_of_interest_'+str(seuil_correl)+'.txt'
	np.savetxt(path_fig+txt_name,particles_filtered_1)
	
	return particles_filtered_1






########################################################################
############# 									           ############# 
#############   FONCTIONS DE CALCULS PHYSIQUES ET DE PLOTS #############      
############# 									           ############# 
########################################################################


############# faire un arrondi à 2 chiffres après la virgule ############# 

def arrondi(x):
	decimal=x-int(x)
	length = len(str(decimal))
	round_x=round(decimal, 2)
	return int(x)+round_x



############# calculer la vitesse de la particule choisie ############# 



def vitesse_particule(file_traj, index, dt=0.01, step=4):
    XYZ = np.array(get_xyz(file_traj, index),dtype=np.float32)
    delta_t = (XYZ[4][step:] - XYZ[4][:-step]) * dt
    N = len(delta_t)
    vitesse=np.zeros(N)
    positions = np.array(XYZ[:3])
    vx=(positions[0][step:]-positions[0][:-step])/delta_t
    vy=(positions[1][step:]-positions[1][:-step])/delta_t
    vz=(positions[2][step:]-positions[2][:-step])/delta_t
    vitesse=np.sqrt(vx**2+vy**2+vz**2)
    
    return vitesse





############# définir une vitesse moyenne glissante ############# 

def moving_average(x, w=5):
    return np.convolve(x, np.ones(w), 'valid') / w
 
 
 
 ############# tracer la vitesse des particules dans le temps ############# 

def plot_speed(traj_file,index,dt1=0.01,new_step=10,save=False,show=False):
	w=new_step
	vitesse_file= vitesse_particule(traj_file,index, dt=dt1,step=new_step)
	# vitesse moyenne
	vm=np.mean(vitesse_file)
	
	# vitesse glissante avec une fenêtre de w=10
	v_mean_moving=moving_average(vitesse_file, w)
	
	# vitesses 3/4 et 1/4
	vstd=np.std(vitesse_file)
	v_min=vm-vstd
	
	if v_min<0:
		v_min=0
	v_max=vm+vstd
	
	N=len(vitesse_file)
	time=np.arange(1,N+1)*dt1
	time_moving_average=np.arange(1,N+2-w)*dt1

	v_mean=[vm]*N
	v_inf=[v_min]*N
	v_sup=[v_max]*N
	
	# plot des graphiques
	plt.figure()

	plt.plot(time,vitesse_file,'*', color='b')
	plt.plot(time,v_mean,'--', color='r',label="v_mean = "+str(int(vm))+"$\mu m . s^{-1}$")
	plt.plot(time,v_sup,'--', color='g' , label=r"$v_{3/4} = $" +str(int(v_max))+"$\mu m . s^{-1}$")
	plt.plot(time_moving_average,v_mean_moving,'--', color='black' , label=r"$v_{moving}$")
	plt.plot(time,v_inf,'--', color='g', label=r"$v_{1/4} = $" +str(int(v_min))+"$\mu m . s^{-1}$")
	
	plt.legend(fontsize=9)
	plt.grid()
	plt.xlabel("temps en seconde")
	plt.ylabel("vitesse des bactéries")
	plt.title("vitesse de sur la trajectoire n°"+str(index))
	
	
	# enregistrement des graphiques
	if save ==True:
		
		path_fig=path + data_folder_name+speed_folder
		figure_name='vitesse_traj_'+str(index)+'.png'
		plt.savefig(path_fig+figure_name,dpi=300,bbox_inches='tight',format='png')
	
	if show==True:
		plt.show()
	plt.close('all')
	return

############# tracer l'histogramme des vitesses de l'ensemble des bactéries #############

def histogram_speed(traj_file,dt=0.01,speed_step=10,save=False,show=False):
	
	list_index=get_good_index(traj_file)
	l=len(list_index)
	v_list_for_all=[]
	
	
	bar = Bar('Processing', max=l)
	for index in list_index:
		t0=time()
		v_index=vitesse_particule(traj_file,index,dt,step=speed_step)
		v_list_for_all+=v_index.tolist()
		t1=time()
		sleep(t1-t0)
		bar.next()
	bar.finish()
	
	plt.hist(v_list_for_all,density = True, bins=len(v_list_for_all)//10)
	plt.xlim(min(v_list_for_all)-10, max(v_list_for_all)+10)
	plt.xlabel(r"vitesse en $\mu m .s^{-1}$")
	plt.ylabel('densité de probabilité')
	plt.title('Histogramme des vitesses des bactéries'+str(speed_step))
		
	
	if save==True:
		path_fig_2=path + data_folder_name+speed_folder
		figure_name='histogram_speed_all_traj'+'.png'
		plt.savefig(path_fig_2+figure_name,dpi=400)
		
	if show==True:
		plt.show()
	
	plt.close('all')

	return 
	
	
	



############# tracer la vitese en fontcion du Tstep ############# 

def speed_time_choice(traj_file,index,dt=0.01,save=False,show=False):
	
	
	t_step=[i for i in range(1,200)]
	vmean=[]

	bar = Bar('Processing', max=len(t_step))
	for time_i in t_step:
		t0=time()
		vitesse_file= vitesse_particule(traj_file,index, dt,step=time_i)
		vm=np.mean(vitesse_file)
		vmean.append(vm)
		t1=time()
		sleep(t1-t0)
		bar.next()
	bar.finish()
		
	# plot des graphiques

	plt.plot(t_step,vmean,'*', color='b')
	plt.grid()
	plt.xlabel("step en ms")
	plt.ylabel("vitesse moyenne des bactéries")
	plt.title("vitesse en fonction du Tstep sur la trajectoire")
	
	
	# enregistrement des graphiques
	if save ==True:
		
		path_fig=path + data_folder_name+speed_folder
		figure_name='vitesse_Tstep_'+str(index)+'.png'
		plt.savefig(path_fig+figure_name,dpi=400)
	
	if show==True:
		plt.show()
	plt.close('all')
	
	


 ############# tracer une trajectoire isolée, en 3D ############# 

def plot_traj(file_traj,index,save=False,show=False):
	XYZ=np.array(get_xyz(file_traj,index),dtype=np.float32)
	
	fig = plt.figure(figsize = (10, 6))
	ax = plt.axes(projection ="3d")
	ax.scatter(XYZ[0],XYZ[1],XYZ[2],marker='o',s=2,color='black')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	ax.axes.set_xlim3d(left=0, right=180) 
	ax.axes.set_ylim3d(bottom=0, top=180) 
	ax.axes.set_zlim3d(bottom=0, top=100) 
	
	if save==True:
		path_fig=path + data_folder_name+traj_folder
		figure_name='trajectoire_'+str(index)+'.png'
		plt.savefig(path_fig+figure_name,dpi=400)
	
	if show==True:
		plt.show()
	plt.close('all')
	return
	
	
	
	
	
 ############# définir une colormap en fonction de la vitesse ############# 	

def color_map(coef):

	c=(128/255,0,1)
	if coef<=0.1:
		c=(0,0,1)
	if 0.1<=coef<0.2:
		c=(0,128/255,1)
	if 0.2<=coef<0.3:
		c=(0,1,1)
	if 0.3<=coef<0.4:
		c=(0,1,128/255)
	if 0.4<=coef<0.5:
		c=(0,1,0)
	if 0.6<=coef<0.7:
		c=(128/255,1,0)
	if 0.7<=coef<0.8:
		c=(1,1,0)
	if 0.8<=coef<0.9:
		c=(1,128/255,0)
	if 0.9<=coef<1:
		c=(1,0,0)
	return c
	




 #############  tracer une trajectoire colorée en fonction de la vitesse des particules ############# 	

def traj_speed(traj_file,index,dt=0.01,save=False,show=False, normalize=False):
	XYZ=np.array(get_xyz(traj_file,index),dtype=np.float32)
	V=vitesse_particule(traj_file,index, dt)
	N=len(XYZ[0])
	X=XYZ[0][2:-2]
	Y=XYZ[1][2:-2]
	Z=XYZ[2][2:-2]
	
	if normalize==True:
		V=abs((V-np.min(V))/(np.max(V)-np.min(V)))

	fig = plt.figure(figsize = (10, 6))
	ax = plt.axes(projection ="3d")
	N=len(V)
	colors=np.array([V[i]*3 for i in range(N)])
	
	ax.scatter(X,Y,Z,s=4,c=colors)
	
	if save ==True:
		path_fig=path + data_folder_name+speed_folder
		figure_name='speed_traj'+str(index)+'.png'
		plt.savefig(path_fig+figure_name,dpi=400)
	
	if show==True:
		plt.show()
	plt.close('all')
	return
	
	
	
	
#############  tracer le volume  de détection par CCL des particules au cours du temps ############# 

def plot_volume(file_traj,index,dt=0.01,save=False,show=False):
	A=np.array(get_xyz(file_traj,index),dtype=np.float32)
	volume=A[3]
	z=A[2]
	N=len(volume)
	time=dt*np.array([*range(1, N+1, 1)])
	plt.plot(z,volume,'*', color='blue')
	plt.axis()
	plt.xlabel('z')
	plt.ylabel('volume')
	plt.title('volume de détection par CCL')
	plt.grid()
	
	if save==True:
		path_fig=path + data_folder_name+volume_folder
		figure_name='volume_'+str(index)+'.png'
		plt.savefig(path_fig+figure_name,dpi=400)
	
	if show==True:
		plt.show()
	plt.close('all')
	return








############# calculer l'angle entre 2 vecteurs en 3D #############  
def get_angle(traj_file,index,step=5):
	xyz=np.array(get_xyz(traj_file,index),dtype=np.float32)
	N=len(xyz[0])
	
	vectors_list=[]
	time_list=[]
	x_pos=xyz[0][step::step]-xyz[0][:-step:step]
	y_pos=xyz[1][step::step]-xyz[1][:-step:step]
	z_pos=xyz[2][step::step]-xyz[2][:-step:step]
	vectors_list=np.array([x_pos,y_pos,z_pos],dtype=np.float32).T
	
	M=len(vectors_list)
	angle_list=np.zeros(M-1)
	time_list=np.zeros(M+1)
	time_list[:]=xyz[4][::step]
	
	
	a=vectors_list[:-1];b=vectors_list[1:]
	product=a*b
	scalar_product=product[:,0]+product[:,1]+product[:,2]
	norm=[]
	for i in range(M-1):
		norm.append(np.linalg.norm(a[i])*np.linalg.norm(b[i]))
	norm=np.array(norm,dtype=np.float32)
	angle_list[:]=np.arccos(scalar_product/norm)*180/np.pi
	
	return angle_list,time_list
	
	

#############  calculer la distribution des angles sur une trajectoire ############# 
#############  obtenir l'évolution des angles des vecteurs vitesses dans le temps ############# 
	
def plot_angle(traj_file,index,step=5,save=False,show=False):
	angle_data=get_angle(traj_file,index,step=5)
	angle_list=angle_data[0]
	time_list=angle_data[1][1:-1]
	
	plt.plot(time_list,angle_list,'--',markersize=3)
	plt.ylabel(r"angle $\theta$ en degré")
	plt.xlabel('temps')
	plt.title('Orientation dans le temps de la trajectoire n°'+str(index))
	
	
	if save==True:
		path_fig_1=path + data_folder_name+angle_folder
		figure_name='angle_time_'+str(index)+'.png'
		plt.savefig(path_fig_1+figure_name,dpi=400)
	
	if show==True:
		plt.show()

	plt.close('all')
	plt.hist(angle_list,density = True, bins=len(angle_list))
		
	plt.xlim(min(angle_list)-10, max(angle_list)+10)
	plt.xlabel(r"angle $\theta$ en degré")
	plt.ylabel('densité de probabilité')
	plt.title('Angle entre les vecteurs vitesses : trajectoire n°'+str(index))
		
	
	if save==True:
		path_fig_2=path + data_folder_name+angle_folder
		figure_name='angle_traj_'+str(index)+'.png'
		plt.savefig(path_fig_2+figure_name,dpi=400)
		
	if show==True:
		plt.show()
	plt.close('all')
	return 
	







########################################################
############# 							   ############# 
#############   FONCTIONS DE POUR LE MSAD  #############      
############# 							   ############# 
########################################################



############# définir les vecteurs pour la MSAD ############# 

def vectors_msad(traj_file,index,vecteur_step=6):
	xyz=get_xyz(traj_file,index)
	N=len(xyz[0])
	vectors_list =[]
	time=xyz[4]
	for i in range(0,N-1-vecteur_step,vecteur_step):
		if time[i+vecteur_step]-time[i]==vecteur_step:
			vector_i=[xyz[0][i+vecteur_step]-xyz[0][i], xyz[1][i+vecteur_step]-xyz[1][i], xyz[2][i+vecteur_step]-xyz[2][i]]
			vectors_list.append(vector_i)
	
	return vectors_list
	
	
	
	
	
	

############# calcul du MSAD : diffusion angulaire ############# 

def angle_msad_i(traj_file,index,vectors_list,step=1):
	
	N=len(vectors_list)
	cos_theta=0
	
	if step<N-1:
		
		for j in range(0,N-1-step,step):
			vector_a=vectors_list[j+step];vector_b=vectors_list[j]
			cos_theta+=np.dot(vector_a,vector_b)/(np.linalg.norm(vector_b)*np.linalg.norm(vector_a))
			
		cos_theta=cos_theta/(N-step)
	return cos_theta
	
	
	
	
	
	
	
	

############# plot du MSAD associé à 1 seule trajectoire ############# 

def plot_angle_MSAD_i(traj_file,index,tmax=800,dt=0.01,save=False,show=False):
	
	msad_list=[]
	time_list=[]
	vectors_list=vectors_msad(traj_file,index,vecteur_step=2)
	

	bar = Bar('Processing', max=tmax)
	for i in range (1,tmax):
		t0=time()
		msad_new=angle_msad_i(traj_file,index,vectors_list,step=i)
		
		if msad_new!=0:
			msad_list.append(msad_new)
			time_list.append(dt*i)	
		t1=time()
		sleep(t1-t0)
		bar.next()
	bar.finish()	

	plt.plot(time_list,msad_list,'*',color='b')
	plt.axis()
	plt.xlabel('temps')
	plt.ylabel('MSAD')
	plt.grid()
	plt.legend()
	plt.title(r'$cos(\theta)$ calculé en fonction du temps : trajectoire '+str(index))
	
	if save==True:
		
		path_fig=path + data_folder_name+angle_folder
		figure_name='angle_MSAD_'+str(index)+'.png'
		txt_name="MSAD_"+str(index)+".txt"
		np.savetxt(path_fig+txt_name,msad_list)
		plt.savefig(path_fig+figure_name,dpi=400)
	
	if show==True:
		plt.yscale('log')
		# ~ plt.xscale('log')

		plt.show()
	plt.close('all')
	return








############# calcul du MSAD sur l'ensemble des trajectoires ############# 

def angle_MSAD_average(traj_file,particles_index,tau=1):
	
	msad_average=0
	
	for index in particles_index:
		vectors_list=vectors_msad(traj_file,index,vecteur_step=4)
		msad_average+=angle_msad_i(traj_file,index,vectors_list,step=tau)

	N_traj=len(particles_index)
	
	msad_average=msad_average/N_traj
	
	return msad_average








############# plot du MSAD moyen ############# 

def plot_MSAD_average(traj_file,particles_index,tmax=800,save=False,show=False):
	
	msad_average_list=[]
	time_list=[]
	dt=0.01

	bar = Bar('Processing', max=tmax)
	for i in range (1,tmax):
		t0=time()
		msad_new=angle_MSAD_average(traj_file,particles_index,tau=i)
		
		if msad_new!=0:
			msad_average_list.append(msad_new)
			time_list.append(dt*i)	
		t1=time()
		sleep(t1-t0)
		bar.next()
	bar.finish()
			

	plt.plot(time_list,msad_average_list,'*',color='b')
	plt.axis()
	plt.xlabel('temps')
	plt.ylabel('MSAD')
	plt.grid()
	plt.legend()
	plt.title(r'$<cos(\theta)>$ moyen calculé en fonction du temps ')
	
	if save==True:
		
		path_fig=path + data_folder_name+angle_folder
		figure_name='MSAD_average'+'.png'
		plt.savefig(path_fig+figure_name,dpi=400)
	
	if show==True:
		plt.show()
	plt.close('all')
	return
	
	
	

	
def get_list_msad(dt=0.01):
	
	path_fig=path + data_folder_name+angle_folder
	name_list=os.listdir(path_fig)
	file_list=[]
	
	for file_name in name_list:
		if file_name.endswith('.txt'):
			file_list.append(file_name)
	
	MSAD_list=[]
	for msad_i in file_list:
		MSAD_list.append(np.loadtxt(path_fig+msad_i))
		
	def trier_list(list_msad):
		N=len(list_msad)
		liste_triee=[]
		Lenght_list=[]
		
		for i in range(N):
			Lenght_list.append([len(list_msad[i]),i])
		Lenght_list.sort()
		
		for j in range(N):
			index=Lenght_list[j][1]
			liste_triee.append(list_msad[index])
		return liste_triee	
	
	MSAD_list=trier_list(MSAD_list)
	
	return MSAD_list
	


def plot_msad_graph_average(dt=0.01,save=False,show=False):
	
	MSAD=get_list_msad()
	MSAD_average=[]; TIME=[]
	N=len(MSAD); Nmin=0; j=0; msad_mean=0
	
	for Nmin in range(N):
		while j<len(MSAD[Nmin]):
			for i in range(Nmin, N):
				msad_mean+=MSAD[i][j]
			msad_mean=msad_mean/(N-Nmin+1)
			MSAD_average.append(msad_mean)
			j+=1
			TIME.append(j*dt)
	
	
	plt.plot(TIME,MSAD_average,'*',color='b')
	plt.xlabel('temps')
	plt.ylabel('<MSAD>')
	plt.title("MSAD moyen calculé via la moyenne des courbes")
	plt.axis()
	plt.grid()
	
	
	if save==True:
		path_fig=path + data_folder_name+angle_folder
		figure_name='MSAD_average_graphic'+'.png'
		txt_name="msad_mean_graphic"+".txt"
		np.savetxt(path_fig+txt_name,MSAD_average)
		plt.savefig(path_fig+figure_name,dpi=400)
	
	if show==True:
		plt.show()
	plt.close('all')
	return




#######################################################
############# 							  ############# 
#############   FONCTIONS DE POUR LE MSD  #############      
############# 							  ############# 
#######################################################



#############  calcul du MSD associé à une particule, puis un ensemble de particule ############# 

def MSD_i(traj_file,index,time_step=1):
	
	xyz=get_xyz(traj_file,index)
	x=xyz[0]; y=xyz[1]; z=xyz[2];time=xyz[4]
	Nj=len(x)
	r2_i=0
	frame_i=0
	
	if time_step<Nj-1:
		for j in range(0,Nj-1-time_step):
			if time[j+time_step]-time[j]==time_step:
				r2_i+=(x[j+time_step]-x[j])**2+(y[j+time_step]-y[j])**2+(z[j+time_step]-z[j])**2
				frame_i+=1
		r2_i=r2_i/(Nj-time_step)
	return r2_i, frame_i






#############  calcul du MSD moyen sur l'ensemble des trajectoires différentes #############  
	
def MSD_average(traj_file,particles_index,t=1):
		
	r2_tot=0
	frame_average=0
	New_list_index=[]	

	for i in range(len(particles_index)) :
		if particles_index[i][1]>t:
			New_list_index.append(particles_index[i][0])
		
	
	for index in New_list_index:
		msd_data=MSD_i(traj_file,index,time_step=t)
		r2_tot+=msd_data[0]
		frame_average+=msd_data[1]
	
	N_traj=len(New_list_index)
	
	r2_tot=r2_tot/N_traj
	return r2_tot,frame_average
	
	
	
	
#############  plot du MSD MOYEN sur toutes les trajectoires #############  
		
def plot_MSD_mean(traj_file,particles_index ,save_plot=False,save_txt=False,show=False):
		
	r2_list=[]
	time_list=[]
	frame_list=[]
	list_index_length=[]
	Nmax=0
	
	for index in particles_index:
		xyz=get_xyz(traj_file,index)
		N_index=len(xyz[0])
		list_index_length.append([index,N_index])
		
		if Nmax<N_index:
			Nmax=N_index
				
	bar = Bar('Processing', max=Nmax)
	for i in range (Nmax):
		t0=time()
		msd_data=MSD_average(traj_file,list_index_length,t=i)
		r2_list.append(msd_data[0])
		frame_list.append(msd_data[1])
		time_list.append(i)
		t1=time()
		sleep(t1-t0)
		bar.next()
	bar.finish()
		
	fig, ax1 = plt.subplots()	
	ax2 = ax1.twinx()
	ax1.plot(time_list,r2_list,'*',color='b',label='MSD')
	ax2.plot(time_list,frame_list,'*',color='r',label=r'$N_{frame}$')
	ax1.set_xlabel(r'Temps (s) ',fontsize=18)
	ax1.set_ylabel(r'MSD', fontsize=18,color='blue')
	ax2.set_ylabel(r'$N_{frame}$', color='red')
	
	ax1.axis()
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.grid()
	
	plt.title('MSD calculé en fonction du temps avec t= '+str(Nmax))
	
	if save_plot==True:
		path_fig=path + data_folder_name+MSD_folder
		figure_name='MSD_average'+str(Nmax)+'.png'
		plt.savefig(path_fig+figure_name,dpi=400)
		
	if save_txt==True:
		path_txt=path + data_folder_name+filter_MSD_txt
		txt_name="msd_mean"+".txt"
		np.savetxt(path_txt+txt_name,r2_list)
	
	if show==True:
		plt.show()
	plt.close('all')
	return
	
	






#############  plot du MSD sur une TRAJECTOIRE UNIQUE #############  
	
def plot_MSD_i(traj_file,index,dt=0.01,folder_txt=filter_MSD_txt,save_plot=False,save_txt=False,show=False):
	
	xyz=get_xyz(traj_file,index)
	tmax=len(xyz[0])
	
	r2_list=[]
	r2_list_modele=[]
	time_list=[]
	time_modele=[]


	n=0
	
	bar = Bar('Processing', max=tmax)
	bar.next()
	
	for i in range (1,tmax):
		
		t0=time()
		msd_data=MSD_i(traj_file,index,time_step=i)
		msd_new=msd_data[0]
		
		if msd_new!=0 :
			r2_list.append(msd_new)
			time_list.append(dt*i)	
			n+=1
			
		t1=time()
		sleep(abs(t1-t0))
		bar.next()
	bar.finish()
			
	plt.plot(time_list,r2_list,'*',color='b')
	plt.axis()
	plt.xscale('log')
	plt.yscale('log')
	plt.grid()
	plt.title('MSD calculé en fonction du temps : trajectoire '+str(index))
	
	if save_plot==True:
		
		path_fig=path + data_folder_name+MSD_folder
		figure_name='MSD_traj'+str(index)+'.png'
		plt.savefig(path_fig+figure_name,dpi=400)
	
	if save_txt==True:
		path_txt=path + data_folder_name+folder_txt
		txt_name="msd"+str(index)+".txt"
		np.savetxt(path_txt+txt_name,r2_list)
		
	if show==True:
		plt.show()
	plt.close('all')
	return
	

	
	
############# plot de TOUS les GRAPHS de MSD ENSEMBLES #############  

def plot_together_MSD(dt=0.01,folder_txt=filter_MSD_txt,save=False,legend=False,show=False):
	
	path_fig=path + data_folder_name+folder_txt
	name_list=os.listdir(path_fig)
	file_list=[]
	
	for file_name in name_list:
		if file_name.endswith('.txt'):
			file_list.append(file_name)
	
	for msd_i in file_list:
		MSD=[]
		TIME=[]
		MSD=np.loadtxt(path_fig+msd_i)
		TIME=np.arange(len(MSD))*dt
		name=msd_i
		name=name.replace('.txt','')
		name=name.replace('msd','')
		plt.plot(TIME,MSD,'*',markersize=3, label='n°'+ str(name))
	
	plt.title("MSD sur l'ensemble des trajectoires des bactéries motiles")
	plt.xlabel('temps')
	plt.ylabel('MSD')
	plt.axis()
	
	if legend==True:
		plt.legend()
	plt.xscale('log')
	plt.yscale('log')
	plt.grid()
	
	if save==True:
		path_fig=path + data_folder_name+MSD_folder
		figure_name='MSD_TOTAL'+'.png'
		plt.savefig(path_fig+figure_name,dpi=400)
	
	if show==True:
		plt.show()
	plt.close('all')
	return
			




############# fit du MSD moyen #############  


def plotfit_MSD(dt=0.01,save=False,show=False):
	
	path_fig=path + data_folder_name+filter_MSD_txt
	name_list=os.listdir(path_fig)
	mds_mean='msd_mean_graphic.txt'
	
	MSD=np.array(np.loadtxt(path_fig+mds_mean),dtype=np.float32)
	MSD_GOOD=[]
	
	def func(t, v2,b):
		return (v2)*t**2
	
	N=0
	for i in range(len(MSD)):
		if MSD[i]!=0:
			MSD_GOOD.append(MSD[i])
			N+=1
			
	TIME=np.arange(N)*dt
	TIME_modele=np.copy(TIME[0:N//5])
	MSD_modele=np.copy(MSD_GOOD[0:N//5])

	
	popt, _ = curve_fit(func, TIME_modele, MSD_modele)
	label_modele= ('modèle :' r'$\Delta r$='+str(int(popt[0]))+r'$t^{2}$'+ '\n'+ r'$V_0$ = ' + str(arrondi(np.sqrt(popt[0]))) +r'$\mu m.s^{-1}$' ) 
	
	
	plt.plot(TIME, func(TIME, *popt), 'k-',color='r',label=label_modele)
	plt.plot(TIME,MSD_GOOD,'*',markersize=3, color='b',label='experience')
	
	plt.xlabel('temps')
	plt.ylabel('MSD average')
	plt.title("MSD moyen l'ensemble des trajectoires des bactéries motiles")
	
	plt.xscale('log')
	plt.yscale('log')
	plt.axis()
	plt.grid()
	plt.legend()

	
	if save==True:
		path_fig=path + data_folder_name+MSD_folder
		figure_name='MSD_fit'+'.png'
		plt.savefig(path_fig+figure_name,dpi=400)
		
	if show==True:
		plt.show()	
	return
	

	



	
############# méthode de calcul graphique du MSD moyen ############# 
############# récupération des données des MSD individuels ############# 
	
def get_list_msd(dt=0.01):
	
	path_fig=path + data_folder_name+filter_MSD_txt
	name_list=os.listdir(path_fig)
	file_list=[]
	
	for file_name in name_list:
		if file_name.endswith('.txt'):
			file_list.append(file_name)
	
	MSD_list=[]
	for msd_i in file_list:
		MSD_list.append(np.loadtxt(path_fig+msd_i))
		
	def trier_list(list_msd):
		N=len(list_msd)
		liste_triee=[]
		Lenght_list=[]
		
		for i in range(N):
			Lenght_list.append([len(list_msd[i]),i])
		Lenght_list.sort()
		
		for j in range(N):
			index=Lenght_list[j][1]
			liste_triee.append(list_msd[index])
		return liste_triee	
	
	MSD_list=trier_list(MSD_list)
	
	return MSD_list
	
	
	
	
############# plot du <MSD> moyen obtenu en moyennant les autres courbes ############# 
	
def plot_msd_graph_average(dt=0.01,save_plot=False,save_txt=False,show=False):
	
	MSD=get_list_msd()
	MSD_average=[]; TIME=[]
	N=len(MSD); Nmin=0; j=0; msd_mean=0
	
	for Nmin in range(N):
		while j<len(MSD[Nmin]):
			for i in range(Nmin, N):
				msd_mean+=MSD[i][j]
			msd_mean=msd_mean/(N-Nmin+1)
			MSD_average.append(msd_mean)
			j+=1
			TIME.append(j*dt)
	
	plt.plot(TIME,MSD_average,'*',color='b')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('temps')
	plt.ylabel('<MSD>')
	plt.title("MSD moyen calculé via la moyenne des courbes")
	plt.axis()
	plt.grid()
	
	if save_plot==True:
		path_fig=path + data_folder_name+MSD_folder
		figure_name='MSD_average_graphic'+'.png'
		plt.savefig(path_fig+figure_name,dpi=400)
	
	if save_txt==True:
		path_txt=path + data_folder_name+filter_MSD_txt
		txt_name="msd_mean_graphic"+".txt"
		np.savetxt(path_txt+txt_name,MSD_average)
	
	if show==True:
		plt.show()
	plt.close('all')
	return

	
	
############# MSD POUR L'ENSEMBLE DES BACTERIES NON MOTILES ############# 

def no_filter_MSD(traj_file,threshold_lenght=400,show=False):
	
	all_index=get_all_index(traj_file)
	filtered_index=get_good_index(traj_file)
	random_index_list=[]
	
	Nmax=len(all_index)
	print("--------------")
	bar = Bar('Loading index of random trajectories', max=Nmax)
	for index in all_index:
		t0=time()
		if index not in filtered_index:
			xyz=np.array(get_xyz(traj_file,index),dtype=np.float32)
			N=len(xyz[0])
			if N>threshold_lenght:
				random_index_list.append(index)
		t1=time()
		sleep(abs(t1-t0))
		bar.next()
	bar.finish()
	
	i=0
	L_random=len(random_index_list)
	for index in random_index_list:
		i+=1
		print("--------------")
		print("Calculating MSD random particle n°"+str(index)+'--'+ str(i)+'/'+str(L_random))
		plot_MSD_i(traj_file,index,folder_txt=MSD_txt_folder,save_txt=True,show=show)
		print("\n")
	return

	
	





#########################################################################
############# 									   			############# 
############# FONCTIONS GLOBALES POUR L'ANALYSE DES DONNEES #############  
############# 									   			############# 
#########################################################################
	
	



############# TRAJECTOIRE, VITESSE, VOLUME ET ANGLE DES PARTICULES ############# 

def apply_for_all(traj_file,save_choice=False,show=False):
	
	list_index=get_good_index(traj_file)

	print("index des trajectoires intéressantes : "+str(list_index))
	
	for particles in list_index:
		
		print("--------------")
		print("particule n°"+str(particles) + "\n")
		
		plot_traj(traj_file,index=particles,save=save_choice,show=show)
		print("trajectoire done")
		
		# ~ traj_speed(traj_file,particles,save=save_choice,show=show)
		# ~ print("trajectoire vitesse done")
		
		# ~ plot_speed(traj_file,particles,new_step=4,save=save_choice,show=show)
		# ~ print("vitesse done")
		
		# ~ plot_volume(traj_file,index=particles,save=save_choice,show=show)
		# ~ print("volume done")
		
		# ~ plot_angle(traj_file,index=particles,save=save_choice,show=show)
		# ~ print("angle done"+"\n")
	
	return






############# MSD INDIVIDUEL, MOYEN ET REGROUPES DES PARTICULES ############# 

def make_MSD_all(traj_file,save_choice=False,show=False):
	
	list_index=get_good_index(traj_file)
	
	
	print("index des trajectoires intéressantes : "+str(list_index))

	for i in list_index:
		print("--------------")
		print("MSD particule n°"+str(i))
		plot_MSD_i(traj_file,index=i,save_plot=save_choice,save_txt=save_choice,show=show)
		print("MSD done"+"\n")
	
	print("--------------")
	plot_msd_graph_average(save_plot=save_choice,save_txt=save_choice,show=show)
	print("\n"+ "MSD graphic average method done"+"\n")
	
	print("--------------")
	print("Calculating MSD average")
	plot_MSD_mean(traj_file,list_index,save_plot=save_choice,save_txt=save_choice,show=show)
	print("\n"+ "MSD calculated average done"+"\n")
	
	print("--------------")
	plot_together_MSD(save=save_choice,legend=False,show=show)
	print("\n"+ "MSD together done"+"\n")
	
	
	plotfit_MSD(save=save_choice,show=show)
	print("\n"+ "MSD average fit done"+"\n")

	return
	

	



############# MSAD INDIVIDUEL, MOYEN ET REGROUPES DES PARTICULES ############# 

def make_MSAD_all(traj_file,save_choice=False,show=False):
	
	list_index=get_good_index(traj_file)
	
	
	print("index des trajectoires intéressantes : "+str(list_index))
	
	for i in list_index:
		print("--------------")
		print("MSAD particule n°"+str(i))
		plot_angle_MSAD_i(traj_file,i,save=save_choice,show=show)
		print("MSAD done"+"\n")
	
	print("--------------")
	print("Calculating MSAD graphic average")
	plot_msad_graph_average(save=save_choice,show=show)
	print("MSAD graphic average done"+"\n")
	
	
	print("--------------")
	print("Calculating MSAD calculated average")
	plot_MSAD_average(traj_file,list_index,save=save_choice,show=show)	
	print("MSAD calculated average done"+"\n")
	return








