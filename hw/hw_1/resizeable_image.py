from imagematrix import ImageMatrix

class ResizeableImage(ImageMatrix):
    def best_seam(self, dp: bool = True) -> list[tuple[int, int]]:
        # TODO: support dp = False.
        min_energy_cache = {}
        energy_cache = {}

        min_energy = None
        min_i = None
        for i in range(self.width):
            energy, _ = self._calc_min_energy(i, self.height-1, min_energy_cache, energy_cache)
            if (min_energy is None) or (energy < min_energy):
                min_energy = energy
                min_i = i

        # Backtrack.
        seam = [(min_i, self.height - 1)]
        for j in reversed(range(self.height - 1)):
            seam.append((min_energy_cache[seam[-1]][1], j))
        return seam

    def _calc_min_energy(self, i: int, j: int, cache: dict, energy_cache: dict) -> tuple[int, int]:
        """
        Calculate the min energy and the previous column 
        of a vertical seam ending at pixel [i,j].
        """        
        if (i,j) in cache:
            return cache[(i,j)]

        min_prev_energy = None
        min_prev_i = None
        prev_j = j - 1
        for prev_i in range(i - 1, i + 2):
            if (prev_i in range(self.width)) and (prev_j in range(self.height)):
                prev_energy = self._calc_min_energy(prev_i, prev_j, cache, energy_cache)[0]
                if (min_prev_energy is None) or (prev_energy < min_prev_energy):
                    min_prev_energy = prev_energy
                    min_prev_i = prev_i
                    

        if (i, j) not in energy_cache:
            energy_cache[(i,j)] = self.energy(i, j)

        min_energy = (min_prev_energy or 0) + energy_cache[(i,j)]
        cache[(i,j)] = (min_energy, min_prev_i)
        return cache[(i,j)]

if __name__ == "__main__":
    img = ResizeableImage("sunset_small.png")
    print(img.best_seam())
