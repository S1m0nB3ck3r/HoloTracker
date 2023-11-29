
int cuda_CCL(unsigned int *d_labels_volume, bool* d_binarised_focus_volume, sizeXYZ volumeSize) {
	int nbIt = 0;
	//init
	ccl_initLabels(d_labels_volume, d_binarised_focus_volume, volumeSize);
	// Resolve the labels
	ccl_resolveLabels(d_labels_volume, volumeSize);
	// Changed Flag
	bool h_changed = true;
	bool *d_changed;
	cudaMalloc((void**)&d_changed, sizeof(bool));
    
	// While labels have changed
	while (h_changed) {
		nbIt++;
		// Copy changed to device
		cudaMemset(d_changed, 0, 1);
		// Label image
		ccl_labelEquivalence(d_labels_volume, d_binarised_focus_volume, d_changed, volumeSize);
		// Copy changed back
		cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
		// Resolve the labels
		ccl_resolveLabels(d_labels_volume, volumeSize);
	}

	//recherche du pixel max et donc du nombre de label
	pixel pixMAx;
	int sizePlan = volumeSize.X * volumeSize.Y;
	unsigned int* d_plan_max;
	cudaMalloc(&d_plan_max, sizeof(unsigned int) * sizePlan);

	for (int p = 0; p < volumeSize.Z; p++) {
		keepMaxPlaneUintDevice(d_plan_max, d_plan_max, &d_labels_volume[p  * sizePlan], sizePlan);
	}

	unsigned int* h_plan_max = new unsigned int[sizePlan];
	cudaMemcpy(h_plan_max, d_plan_max, sizeof(unsigned int)*sizePlan, cudaMemcpyDeviceToHost);
	unsigned int labelMax = 0;
	unsigned int xyLabel = 0;

	for (int x = 0; x < volumeSize.X; x++) {
		for (int y = 0; y < volumeSize.Y; y++) {
			xyLabel = h_plan_max[x + y * volumeSize.X];
			labelMax = (xyLabel > labelMax) ? xyLabel : labelMax;
		}
	}
	delete(h_plan_max);
	cudaFree(d_plan_max);

	return(labelMax);
}



int cuda_CCA(unsigned int *d_labels_volume, sizeXYZ volumeSize, cublasHandle_t handleCublas, vector <objet> &listeObjets) {

	int nthread = 1024;
	int nblock = volumeSize.X * volumeSize.Y / nthread + 1;

	//reinitialisation liste objets
	listeObjets.clear();
	objet objetNull = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	//recherche du pixel max et donc du nombre de label
	pixel pixMAx;
	int sizePlan = volumeSize.X * volumeSize.Y;
	unsigned int* d_plan_max;
	cudaMalloc(&d_plan_max, sizeof(unsigned int) * sizePlan);

	for (int p = 0; p < volumeSize.Z; p++) {
		keepMaxPlaneUintDevice(d_plan_max, d_plan_max, &d_labels_volume[p  * sizePlan], sizePlan);
	}

	unsigned int* h_plan_max = new unsigned int[sizePlan];
	cudaMemcpy(h_plan_max, d_plan_max, sizeof(unsigned int)*sizePlan, cudaMemcpyDeviceToHost);
	unsigned int labelMax = 0;
	unsigned int xyLabel = 0;

	for (int x = 0; x < volumeSize.X; x++) {
		for (int y = 0; y < volumeSize.Y; y++) {
			xyLabel = h_plan_max[x + y * volumeSize.X];
			labelMax = (xyLabel > labelMax) ? xyLabel : labelMax;
		}
	}
	delete(h_plan_max);
	cudaFree(d_plan_max);

	//analyseLabelPlane
	//allocation listeObjet et initialisation
	objet* h_tabObjets = (objet*)malloc(sizeof(objet)*labelMax);
	for (int o = 0; o <= labelMax; o++) {
		h_tabObjets[o] = objetNull;
	}
	objet* d_tabObjets;
	cudaMalloc(&d_tabObjets, sizeof(objet)*labelMax);

	int labelCount;

	ccl_analyseLabelToObjets(d_labels_volume, d_tabObjets, &labelCount, volumeSize);

	//copie resultat vers host
	cudaMemcpy(h_tabObjets, d_tabObjets, sizeof(objet)*labelMax, cudaMemcpyDeviceToHost);
	for (int i = 0; i < labelMax; i++) {
		listeObjets.push_back(h_tabObjets[i]);
	}
	//allocation des objets à analyser
	free(h_tabObjets);
	cudaFree(d_tabObjets);
	return(0);
}

