# -*- coding: utf-8 -*-

import cupy as cp
from typeHolo import *
loaded_from_source = r'''

extern "C"{

typedef struct {
    double re, im;
}MYCOMPLEX;

typedef struct {
	int X, Y, Z;
}sizeXYZ, coordXYZ;

typedef struct {
	int X, Y;
}sizeXY, coordXY;   


//------------------------------------------------------------------------------------------------------------------------
// Device Functions
//------------------------------------------------------------------------------------------------------------------------

// ---------- Find the root of a chain ----------
__device__ __inline__ unsigned int find_root(unsigned int *labels, unsigned int label) {
	// Resolve Label
	unsigned int next = labels[label];

	// Follow chain
	while(label != next) {
		// Move to next
		label = next;
		next = labels[label];
	}

	// Return label
	return label;
}

__global__ void cuda_Binaries_Focus_Volume(bool* d_bin_volume, double* d_volume, double threshold, unsigned long long size) {

	unsigned long long ijk = blockDim.x * blockIdx.x + threadIdx.x;

	if (ijk < size) {
		d_bin_volume[ijk] = (d_volume[ijk] > threshold) ? 0 : 1;;
	}
}

// Initialise Kernel
__global__ void device_init_labels(unsigned int* d_labels_volume, bool* d_binarised_focus_volume, int sizeX, int sizeY, int sizeZ) {

	// Calculate index
	const int ijk = threadIdx.x + blockDim.x* blockIdx.x;

	int planSize = sizeX * sizeY;
	int nbPix = planSize * sizeZ;

	if (ijk >= nbPix) {
		return;
	}

	int posZ = ijk / planSize;
	int posY = (ijk - posZ * planSize) / sizeX;
	int posX = ijk - posY * sizeX - posZ * planSize;

	// Check Range
	if((posX < sizeX) && (posY < sizeY) && (posZ < sizeZ)) {
		// Load image
		const bool bin_ijk = d_binarised_focus_volume[ijk];

		// Neighbour Connections
		const bool nzm1yx   = (posZ > 0) ? (bin_ijk == d_binarised_focus_volume[(posZ -1) * planSize + posY * sizeX + posX]) : false;
		const bool nzym1x   = (posY > 0) ? (bin_ijk == d_binarised_focus_volume[posZ * planSize + (posY -1) * sizeX + posX]) : false;
		const bool nzyxm1   = (posX > 0) ? (bin_ijk == d_binarised_focus_volume[posZ * planSize + posY * sizeX + posX -1]) : false;

		// Label
		unsigned int label = 0;

		// Initialise Label
		label = (nzyxm1) ? (posZ * planSize + posY * sizeX + posX - 1) : (posZ * planSize + posY * sizeX + posX);
		label = (nzym1x) ? (posZ * planSize + (posY - 1) * sizeX + posX) : label;
		label = (nzm1yx) ? ((posZ - 1) * planSize + posY * sizeX + posX) : label;

		// Write to Global Memory
		d_labels_volume[ijk] = label;
	}
}


// Resolve Kernel
__global__ void device_resolve_labels(unsigned int* d_labels_volume, int sizeX, int sizeY, int sizeZ) {
	
	// Calculate index
	const int ijk = threadIdx.x + blockDim.x* blockIdx.x;

	int nbPix = sizeX * sizeY * sizeZ;

	// Check Thread Range
	if(ijk < nbPix) {
		// Resolve Label
		d_labels_volume[ijk] = find_root(d_labels_volume, d_labels_volume[ijk]);
	}
}


__global__ void device_label_equivalence(unsigned int *d_labels_volume, bool* d_binarised_focus_volume, bool *changed,int sizeX, int sizeY, int sizeZ) {
	// Calculate index
	const int ijk = threadIdx.x + blockDim.x* blockIdx.x;

	int planSize = sizeX * sizeY;
	int nbPix = planSize * sizeZ;

	if (ijk > nbPix) {
		return;
	}

	int posZ = ijk / planSize;
	int posY = (ijk - posZ * planSize) / sizeX;
	int posX = ijk - posY * sizeX - posZ * planSize;

	// Check Range
	if((posX < sizeX) && (posY < sizeY) && (posZ < sizeZ)) {
		// Get image and label values
		const bool bin_ijk = d_binarised_focus_volume[ijk];
		
		// Neighbouring indexes
		const unsigned int xm1 = posX -1;
		const unsigned int xp1 = posX +1;
		const unsigned int ym1 = posY -1;
		const unsigned int yp1 = posY +1;
		const unsigned int zm1 = posZ -1;
		const unsigned int zp1 = posZ +1;
		
		// Get neighbour labels
		const unsigned int lzm1yx = (posZ > 0) ? d_labels_volume[zm1 * planSize +  posY * sizeX +  posX] : 0;
		const unsigned int lzym1x = (posY > 0) ? d_labels_volume[posZ *planSize + ym1 * sizeX +  posX] : 0;
		const unsigned int lzyxm1 = (posX > 0) ? d_labels_volume[posZ * planSize +  posY * sizeX + xm1] : 0;
		const unsigned int lzyx   = d_labels_volume[posZ * planSize +  posY * sizeX +  posX];
		const unsigned int lzyxp1 = (posX < sizeX-1) ? d_labels_volume[posZ * planSize +  posY * sizeX + xp1] : 0;
		const unsigned int lzyp1x = (posY < sizeY -1) ? d_labels_volume[posZ * planSize + yp1 * sizeX +  posX] : 0;
		const unsigned int lzp1yx = (posZ < sizeZ -1) ? d_labels_volume[zp1 * planSize +  posY * sizeX +  posX] : 0;

		//Get neighbour values
		const bool nzm1yx = (posZ > 0)    ? (bin_ijk == d_binarised_focus_volume[zm1 * planSize + posY * sizeX + posX]) : false;
		const bool nzym1x = (posY > 0)    ? (bin_ijk == d_binarised_focus_volume[posZ *planSize + ym1 * sizeX + posX]) : false;
		const bool nzyxm1 = (posX > 0)    ? (bin_ijk == d_binarised_focus_volume[posZ * planSize + posY * sizeX + xm1]) : false;
		const bool nzyxp1 = (posX < sizeX-1) ? (bin_ijk == d_binarised_focus_volume[posZ * planSize + posY * sizeX + xp1]) : false;
		const bool nzyp1x = (posY < sizeY-1) ? (bin_ijk == d_binarised_focus_volume[posZ * planSize + yp1 * sizeX + posX]) : false;
		const bool nzp1yx = (posZ < sizeZ-1) ? (bin_ijk == d_binarised_focus_volume[zp1 * planSize + posY * sizeX + posX]) : false;
		
		// Lowest label
		unsigned int label = lzyx;

		// Find lowest neighbouring label
		label = ((nzm1yx) && (lzm1yx < label)) ? lzm1yx : label;
		label = ((nzym1x) && (lzym1x < label)) ? lzym1x : label;
		label = ((nzyxm1) && (lzyxm1 < label)) ? lzyxm1 : label;
		label = ((nzyxp1) && (lzyxp1 < label)) ? lzyxp1 : label;
		label = ((nzyp1x) && (lzyp1x < label)) ? lzyp1x : label;
		label = ((nzp1yx) && (lzp1yx < label)) ? lzp1yx : label;



		// If labels are different, resolve them
		if(label < lzyx) {
			// Update label
			// Nonatomic write may overwrite another label but on average seems to give faster results
			d_labels_volume[lzyx] = label;

			// Record the change
			changed[0] = true;
		}
	}
}

//analyse label plane to determine objets
__global__ void device_analyseLabelToObjets(unsigned int *d_labels_volume, objet* objets, sizeXYZ volumeSize) {

	int volumeSizeXYZ = volumeSize.X * volumeSize.Y * volumeSize.Z;
	int planSizeXY = volumeSize.X * volumeSize.Y;

	int ijk = threadIdx.x + blockDim.x* blockIdx.x;
	if (ijk > volumeSizeXYZ) {
		return;
	}

	int posZ = ijk / planSizeXY;
	int posY = (ijk - posZ * planSizeXY) / volumeSize.X;
	int posX = ijk - posZ * planSizeXY - posY * volumeSize.X);

	unsigned int label;
	
	if (d_labels_volume[ijk] != 0) {
		label = d_labels_volume[ijk];

		if (objets[label].nbPix == 0) {
			//1er pix dans objet
			atomicExch(&(objets[label].xMin), posX);
			atomicExch(&(objets[label].xMax), posX);
			atomicExch(&(objets[label].yMin), posY);
			atomicExch(&(objets[label].yMax), posY);
			atomicExch(&(objets[label].zMin), posZ);
			atomicExch(&(objets[label].zMax), posZ);
		}
		else {
			if (posX < objets[label].xMin) { atomicExch(&(objets[label].xMin), posX); }
			if (posX > objets[label].xMax) { atomicExch(&(objets[label].xMax), posX); }

			if (posY < objets[label].yMin) { atomicExch(&(objets[label].yMin), posY); }
			if (posY > objets[label].yMax) { atomicExch(&(objets[label].yMax), posY); }

			if (posZ < objets[label].zMin) { atomicExch(&(objets[label].zMin), posZ); }
			if (posZ > objets[label].zMax) { atomicExch(&(objets[label].zMax), posZ); }

		}

		//incr nb pix
		atomicAdd(&(objets[label].nbPix), 1);

		//incr sumX Y Z
		atomicAdd(&(objets[label].pSumX), 1);
		atomicAdd(&(objets[label].pSumY), 1);
		atomicAdd(&(objets[label].pSumZ), 1);

		//incr p x sumX Y Z
		atomicAdd(&(objets[label].pSumX), posX);
		atomicAdd(&(objets[label].pSumY), posY);
		atomicAdd(&(objets[label].pSumZ), posZ);

	}
	

}




}'''

#__global__ void cuda_Binaries_Focus_Volume(bool* d_bin_volume, double* d_volume, double threshold, long volumeSize) {
#__global__ void device_init_labels(unsigned int* d_labels_volume, bool* d_binarised_focus_volume, sizeXYZ volumeSize) {
#__global__ void device_resolve_labels(unsigned int* d_labels_volume, sizeXYZ volumeSize) {
#__global__ void device_label_equivalence(unsigned int *d_labels_volume, bool* d_binarised_focus_volume, bool *changed, sizeXYZ volumeSize) {



module = cp.RawModule(code=loaded_from_source)
ker_cuda_Binaries_Focus_Volume = module.get_function('cuda_Binaries_Focus_Volume')
ker_device_init_labels = module.get_function('device_init_labels')
ker_device_resolve_labels = module.get_function('device_resolve_labels')
ker_device_label_equivalence = module.get_function('device_label_equivalence')

def CCL(d_bin_volume, d_labels_volume, d_volume, threshold, sizeX, sizeY, sizeZ):
	
    n_threads = 1024
    n_blocks = (sizeX * sizeY * sizeZ) // n_threads + 1
    nb_iteration = 0

    # binaries the volume
    ker_cuda_Binaries_Focus_Volume(
        (n_blocks,), (n_threads,), (d_bin_volume, d_volume, threshold, sizeX, sizeY, sizeZ))

    # init labels
    ker_device_init_labels((n_blocks,), (n_threads,),
                           (d_labels_volume, d_bin_volume, sizeX, sizeY, sizeZ))

    # resolve the labels
    ker_device_resolve_labels(
        (n_blocks,), (n_threads,), (d_labels_volume, sizeX, sizeY, sizeZ))
	
    d_changed = cp.bool8_(1)

    while d_changed:
        nb_iteration = nb_iteration + 1
        d_changed = False
        ker_device_label_equivalence(
				(n_blocks,), (n_threads,), (d_labels_volume, d_bin_volume, d_changed, sizeX, sizeY, sizeZ))
        ker_device_resolve_labels((n_blocks,), (n_threads,)(
				d_labels_volume, sizeX, sizeY, sizeZ))

	#recherche du pix max
    return cp.max(d_labels_volume)


def CCA(d_labels_volume, d_volume, threshold, sizeX, sizeY, sizeZ):
	
	xyzSize = sizeX * sizeY * sizeZ

	h_labels_volume = cp.asnumpy(d_labels_volume)
	h_focus_volume = cp.asnumpy(d_volume)

	bacterie = objet()
	for x in range(sizeX):
		for y in range(sizeY):
			for z in range(sizeZ):



""" analyseObjects() {

	unsigned int xyzSize = infoHologramme.imageHeight * infoHologramme.imageWidth * infoHologramme.nb_plan;
	int xySize = infoHologramme.imageHeight * infoHologramme.imageWidth;
	int xSize = infoHologramme.imageWidth;
	listeObjets.clear();

	unsigned int* h_labels_volume = (unsigned int*)malloc(sizeof(unsigned int)*xyzSize);
	cudaMemcpy(h_labels_volume, d_labels_volume, sizeof(unsigned int)*xyzSize, cudaMemcpyDeviceToHost);

	double* h_focus_volume = (double*)malloc(sizeof(double)*xyzSize);
	cudaMemcpy(h_focus_volume, d_focus_volume, sizeof(double)*xyzSize, cudaMemcpyDeviceToHost);


	unsigned int label;
	int indexLabelTrouve;

	objet newObj = { 0,0,0,0,0,0,0,0,0,0,0,0,0.0,0.0,0.0 };
	int posX, posY, posZ, posXYZ;

	for (int k = 0; k < infoHologramme.nb_plan; k++) {
		//cout << "X = " << i << endl;
		for (int j = 0; j < infoHologramme.imageHeight; j++) {
			for (int i = 0; i < infoHologramme.imageWidth; i++) {
				posXYZ = i + j * xSize + k * xySize;
				//quel est le label?
				label = h_labels_volume[posXYZ];
				if (label != 0) {
					//est ce que le label est déjà dans la liste des ojets trouvés
					indexLabelTrouve = labelIsInListeObjets(listeObjets, label);
					if (indexLabelTrouve != 0) {
						indexLabelTrouve--;
						//si oui, on met à jour l'objet
						updateObjet(listeObjets, indexLabelTrouve, { i,j,k }, h_focus_volume[posXYZ]);
						//cout << "objet mis à jour label:" << listeObjets[indexLabelTrouve].label << endl;

					}
					else {
						//sinon on ajoute un nouvel objet à la liste d'objet
						newObj.label = label;
						newObj.nbPix = 1;
						newObj.xMin = i;
						newObj.xMax = i;
						newObj.yMin = j;
						newObj.yMax = j;
						newObj.zMin = k;
						newObj.zMax = k;
						newObj.pSum = h_focus_volume[posXYZ];
						newObj.pxSumX = i * h_focus_volume[posXYZ];
						newObj.pxSumY = j * h_focus_volume[posXYZ];
						newObj.pxSumZ = k * h_focus_volume[posXYZ];

						listeObjets.push_back(newObj);
						//cout << "objet ajouté label:" << newObj.label << endl;
					}
				}
			}
		}
	}
	
	//on calcul les barycentres
	for (int obj = 0; obj < listeObjets.size(); obj++){
		listeObjets[obj].baryX = double(listeObjets[obj].pxSumX) / double(listeObjets[obj].pSum);
		listeObjets[obj].baryY = double(listeObjets[obj].pxSumY) / double(listeObjets[obj].pSum);
		listeObjets[obj].baryZ = double(listeObjets[obj].pxSumZ) / double(listeObjets[obj].pSum);
	}

	
	free(h_labels_volume);
	free(h_focus_volume);

	return(listeObjets.size());
} """
