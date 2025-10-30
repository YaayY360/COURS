/*
 * utils.c
 *
 *  Created on: Oct 29, 2025
 *      Author: y360
 */
#define ARRAY_SIZE 120
#include "utils.h"

int arrayTest[ARRAY_SIZE]={0, 1, 5, 10, -1, 0, 0, 0, 0, 12, 1024, 10, 45, 6, 4,0, 1, 5, 10, -1, 0, 0, 0, 0, 12,
		1024, 10, 45, 6, 4,0, 1, 5, 10, -1, 0, 0, 0, 0, 12, 1024, 10, 45, 6, 4,
		0, 1, 5, 10, -1, 0, 0, 0, 0, 12, 1024, 10, 45, 6, 4,
		0, 1, 5, 10, -1, 0, 0, 0, 0, 12, 1024, 10, 45, 6, 4,
		0, 1, 5, 10, -1, 0, 0, 0, 0, 12, 1024, 10, 45, 6, 4,
		0, 1, 5, 10, -1, 0, 0, 0, 0, 12, 1024, 10, 45, 6, 4,
		0, 1, 5, 10, -1, 0, 0, 0, 0, 12, 1024, 10, 45, 6, 4};



void setup(){
	printf("le système a démarré");
}


void loop(){
	long start=HAL_GetTick();
	Bubble(arrayTest,ARRAY_SIZE);
	long stop=HAL_GetTick();
	printf("Solved in : %d milliseconds \n\r", stop-start);
}


int getMinPosition(int * array, int * toBeIgnored, int size) {
	int min=array[0];
	int imin=0;
	for (int i=0; i<size; i++) {
		if (toBeIgnored[i]!=1) {
			if (array[i]<min){
				imin=i;
				min=array[i];
			}
		}
	}
	return imin;
}


int sortArray(int * toSort, int * sorted, int size){
	int sortedPosition[ARRAY_SIZE]={0};
	for (int i=0; i<size;i++){
		int indexmin=getMinPosition(toSort, sortedPosition, ARRAY_SIZE);
		sortedPosition[indexmin]=1;
		sorted[i]=toSort[indexmin];
	}
	return sorted;

}

int Bubble(int * array, int size){
	for (int i =size-1 ; i > 0 ; i--){
		for (int j =0 ; j<i-1; j++){
			if (array[j+1]<array[j]){
				int b=array[j+1];
				int a= array[j];
				array[j]=b;
				array[j+1]=a;
			}

		}
	}
	return array;
}

