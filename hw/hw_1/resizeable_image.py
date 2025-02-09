from imagematrix import ImageMatrix
import numpy as np
import time

class ResizeableImage(ImageMatrix):    
    def best_seam(self, dp: bool = True) -> list[tuple[int, int]]:
        if dp:
            return self._best_seam_dp()
        return self._best_seam_recur()

    def _best_seam_dp(self) -> list[tuple[int, int]]:
        """Compute the best seam using dynamic programming."""
        # Assumes non-neg int energy. Use a dict if this is not true.
        min_energy_table = np.full((self.width, self.height), -1, dtype=np.int32)
        min_ind_table = np.empty_like(min_energy_table)
        energy_cache = min_energy_table.copy()

        # Base case.
        for i in range(self.width):
            min_energy_table[i, 0] = self._get_cached_energy(i, 0, energy_cache)

        # Fill in the rest of the table using the recursive relation.
        for j in range(1, self.height):
            for i in range(self.width):
                if min_energy_table[i,j] == -1:
                    self._calc_min_energy_dp(i, j, min_energy_table, min_ind_table, energy_cache)
        
        min_i = min_energy_table[:, -1].argmin()

        # Backtrack.
        seam = [(min_i, self.height - 1)]
        for j in reversed(range(self.height - 1)):
            seam.append((min_ind_table[seam[-1]], j))
        return seam
    
    def _get_cached_energy(self, i: int, j: int, energy_cache: np.ndarray) -> int:
        """Returns the cached or calculated energy of the pixel [i,j]."""
        if energy_cache[i, j] == -1:
            energy_cache[i,j] = self.energy(i, j)
        return energy_cache[i,j]

    def _calc_min_energy_dp(self, i: int, j: int, min_energy_table: np.ndarray, min_ind_table: np.ndarray, energy_cache: np.ndarray) -> None:
        """Calculate the min energy and the previous column of a vertical seam ending at pixel [i, j] using DP."""
        prev_j = j - 1
        left_i, right_i = max(0, i - 1), min(i + 2, self.width)
        min_prev_i = min(range(left_i, right_i), key=lambda idx: min_energy_table[idx, prev_j])

        min_energy_table[i, j] = min_energy_table[min_prev_i, prev_j] + self._get_cached_energy(i, j, energy_cache)
        min_ind_table[i, j] = min_prev_i
    
    def _best_seam_recur(self) -> list[tuple[int, int]]:
        """Compute the best seam using naive recursion."""
        energy_cache = np.full((self.width, self.height), -1, dtype=np.int32) 
        min_energy = float("inf")
        best_seam = None
        for i in range(self.width):
            energy, seam = self._calc_min_energy_recur(i, self.height - 1, energy_cache)
            if energy < min_energy:
                min_energy = energy
                best_seam = seam
        return best_seam
    
    def _calc_min_energy_recur(self, i: int, j: int, energy_cache: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
        """Calculate the min energy and the path of the best vertical seam ending at pixel [i,j] recursively."""
        if j == 0:
            return self._get_cached_energy(i, j, energy_cache), [(i,j)]

        min_energy = float("inf")
        best_seam = None
        for prev_i in range(max(0, i-1), min(i+2, self.width)):
            energy, seam = self._calc_min_energy_recur(prev_i, j-1, energy_cache)
            if energy < min_energy:
                min_energy = energy
                best_seam = seam
        min_energy += self._get_cached_energy(i, j, energy_cache)
        return min_energy, [(i,j)] + best_seam 
        
if __name__ == "__main__":
    img = ResizeableImage("sunset_full.png")

    t0 = time.time()
    for i in range(100):
        img.best_seam()
    print(time.time() - t0)
