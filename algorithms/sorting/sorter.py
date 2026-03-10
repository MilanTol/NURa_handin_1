import numpy as np


class Sorter:

    def __init__(self, arr: np.ndarray, make_indx:bool =True):
        """
        initializing sorter object.

            arr (np.ndarray): array to be sorted (using < operator)
            make_indx (bool): whether to keep create an indexing array or not
        """
        self.arr = arr.copy()
        self.make_indx = make_indx
        if make_indx:
            self.indx = np.arange(len(arr))
        

    def selection_sort(self):
        """
        Sorts an array from smallest to largest (using < operator).
        """
        sorted_arr = self.arr.copy()
        N = len(sorted_arr)
        for i in range(N-1):
            i_min = i
            for j in range(i+1, N, 1):
                if sorted_arr[j] < sorted_arr[i_min]:
                    i_min = j
            if i_min != i:
                sorted_arr[[i, i_min]] = sorted_arr[[i_min, i]] #swap element at index i with minimum element (at index i_min)
                if self.make_indx:
                    self.indx[[i, i_min]] = self.indx[[i_min, i]]
        self.sorted_arr = sorted_arr
        return sorted_arr
    

    def quicksort(self, arr: np.ndarray = None):
        """
        Sorts the array stored in Sorter object.
        If passed an array, sorts the array (Does not make a copy!)
        """
        if arr is None:
            arr = self.arr.copy()
        else:
            pass

        N = len(arr)

        if N == 1:
            return arr
        if N == 2:
            if arr[0] > arr[1]:
                arr[[0,1]] = arr[[0,1]]
            return arr
        
        #set pivot index to halfway
        pivot_idx = N//2

        # order the first, middle and last element
        if arr[pivot_idx] > arr[-1]:
            arr[[pivot_idx, -1]] = arr[[-1, pivot_idx]]

        if arr[0] > arr[pivot_idx]:
            arr[[0, pivot_idx]] = arr[[pivot_idx, 0]]

        if arr[pivot_idx] > arr[-1]:
            arr[[pivot_idx, -1]] = arr[[-1, pivot_idx]]

        # set middle to pivot
        pivot = arr[pivot_idx]

        i = 0
        i_locked = False
        j = N - 1
        j_locked = False
        while i < j:
            if not i_locked: #if i is not locked
                if i < pivot_idx: 
                    i += 1 # increase it, if it is not the pivot.
                if arr[i] >= pivot: # check whether it supercedes the pivot
                    i_locked = True # lock in as next candidate to be swapped

            if not j_locked: 
                if j > pivot_idx:
                    j -= 1 # decrease it!
                if arr[j] <= pivot:
                    j_locked = True
            
            if i_locked and j_locked: # if 2 candidates are selected:
                arr[[i, j]] = arr[[j, i]] #swap i and j entries
                i_locked = False # release locks from i and j
                j_locked = False 
                    
                if i == pivot_idx: #change the pivot index accordingly if it has shifted
                    pivot_idx = j
                elif j == pivot_idx:
                    pivot_idx = i

        arr[:pivot_idx] = self.quicksort(arr[:pivot_idx]) #order the subarrays
        arr[pivot_idx:] = self.quicksort(arr[pivot_idx:])
        
        self.sorted_arr = arr #store the sorted array
        return arr


            


        