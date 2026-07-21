import os
from os.path import join as pjoin
from datetime import datetime
from collections.abc import Sequence

import yaml
import numpy as np
import cv2 as cv
from scipy.optimize import linear_sum_assignment

from lpf.utils import get_hash_digest


class EvoSearch:
    def __init__(self,
                 config=None,
                 model=None,
                 solver=None,
                 converter=None,
                 targets=None,
                 objectives=None,
                 droot_output=None,
                 spot_weight=0.0,
                 spot_min_area=50,
                 spot_img_size=128):
        
        self.config = config
        self.model = model
        self.solver = solver
        self.converter = converter

        if isinstance(targets, Sequence) and len(targets) < 1:
            raise ValueError("targets should be a sequence, "\
                             "which must have at least one target.")

        self.targets = targets

        self.objectives = objectives

        self.spot_weight = spot_weight
        self.spot_min_area = spot_min_area
        self.spot_img_size = spot_img_size

        self.bounds_min, self.bounds_max = self.model.get_param_bounds()

        # Create a cache using dict.
        self.cache = {}

        # Create output directories.
        str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.dpath_output = pjoin(droot_output, "search_%s"%(str_now))        
        self.dpath_population = pjoin(self.dpath_output, "population")
        self.dpath_best = pjoin(self.dpath_output, "best")

        os.makedirs(self.dpath_output, exist_ok=True)
        os.makedirs(self.dpath_population, exist_ok=True)
        os.makedirs(self.dpath_best, exist_ok=True)        
        
        # Write the config file.
        fpath_config = pjoin(self.dpath_output, "config.yaml")
        with open(fpath_config, 'wt') as fout:
            yaml.dump(config, fout, default_flow_style=False)
    
    # ================================================================
    # Spot position scoring (optional; requires lpf[spot-scoring])
    # ================================================================

    def _get_centroids(self, img):
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                "spot_weight > 0 requires opencv-python. "
                "Install it via: pip install lpf[spot-scoring]"
            ) from e

        arr = np.array(img)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
        mask = (mask1 | mask2).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        centroids = []
        for s in stats[1:]:
            if s[cv2.CC_STAT_AREA] > self.spot_min_area:
                cx = s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH] / 2
                cy = s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT] / 2
                centroids.append([cx, cy])
        return np.array(centroids) if centroids else np.empty((0, 2))

    def _spot_position_score(self, morph_img, target_img):
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as e:
            raise ImportError(
                "spot_weight > 0 requires scipy. "
                "Install it via: pip install lpf[spot-scoring]"
            ) from e

        c_morph = self._get_centroids(morph_img)
        c_target = self._get_centroids(target_img)
        n_m, n_t = len(c_morph), len(c_target)

        count_penalty = abs(n_m - n_t) / max(n_t, 1)
        if n_m == 0 or n_t == 0:
            return count_penalty

        cost = np.linalg.norm(c_morph[:, None] - c_target[None, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        position_penalty = cost[row_ind, col_ind].mean() / self.spot_img_size

        return count_penalty + position_penalty

    def fitness(self, x):
        digest = get_hash_digest(x)

        if digest in self.cache:
            arr_color = self.cache[digest]
        else:
            x = x[None, :]
            initializer = self.converter.to_initializer(x)
            params = self.converter.to_params(x)

            self.model.initializer = initializer
            self.model.params = params

            # Check constraints and ignore the decision vector if it does not satisfy.
            # if not self.model.check_constraints():
            #    return [np.inf]

            try:
                self.solver.solve(self.model)
            except (ValueError, FloatingPointError) as err:
                print("[ERROR IN FITNESS EVALUATION]", err)
                return [np.inf]

            # idx = self.model.u > self.model.thr
            #
            # if not idx.any():
            #     return [np.inf]
            # elif self.model.u.size == idx.sum():
            #     return [np.inf]
            # elif np.allclose(self.model.u[idx], self.model.u[idx].mean()):
            #     return [np.inf]

            # Colorize the morph model.
            arr_color = self.model.colorize()

            # Store the colorized object in the cache.
            self.cache[digest] = arr_color
        # end of if-else

        # Evaluate objectives.
        morph, pattern = self.model.create_image(0, arr_color)
        morph_rgb = morph.convert("RGB")
        imgs = [morph.convert("RGB")]
        sum_obj = 0.0
        for obj in self.objectives:
            val = obj.compute(imgs, self.targets)
            sum_obj += float(np.sum(val))

        # Spot position score (only computed if enabled).
        if self.spot_weight > 0.0:
            spot_scores = [self._spot_position_score(morph_rgb, t) for t in self.targets]
            sum_obj += self.spot_weight * float(np.mean(spot_scores))

        return [sum_obj]

    def get_bounds(self):
        lb = self.bounds_min.get() if hasattr(self.bounds_min, 'get') else self.bounds_min
        ub = self.bounds_max.get() if hasattr(self.bounds_max, 'get') else self.bounds_max
        return (lb, ub)

    # def save(self, 
    #          mode,
    #          dv,             
    #          max_generation=None,
    #          generation=None,
    #          fitness=None,
    #          arr_color=None):

    #     dv = dv[None, :]

    #     self.model.initializer = self.converter.to_initializer(dv)
    #     self.model.params = self.converter.to_params(dv)

    #     str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
        
    #     if generation is None:
    #         str_gen = ""
    #     else:
    #         if max_generation is None:
    #             max_generation = 1000000
                
    #         fstr_gen = "gen-%0{}d_".format(int(np.ceil(np.log10(max_generation)))+1)
    #         str_gen = fstr_gen%(int(generation))
        
    #     if mode == "pop":
    #         fpath_model = pjoin(self.dpath_population,
    #                             "%smodel_%s.json"%(str_gen, str_now))    

    #         fpath_morph = pjoin(self.dpath_population,
    #                             "%smorph_%s.png"%(str_gen, str_now))

    #         fpath_pattern = pjoin(self.dpath_population,
    #                               "%spattern_%s.png"%(str_gen, str_now))
            
    #     elif mode == "best":            
    #         fpath_model = pjoin(self.dpath_best,
    #                             "%smodel_%s.json"%(str_gen, str_now))
            
    #         fpath_morph = pjoin(self.dpath_best,
    #                             "%smorph_%s.png"%(str_gen, str_now))

    #         fpath_pattern = pjoin(self.dpath_best,
    #                               "%spattern_%s.png"%(str_gen, str_now))
    #     else:
    #         raise ValueError("mode should be 'pop' or 'best'")

    #     if arr_color is None:
    #         digest = get_hash_digest(dv)            
    #         if digest not in self.cache:                
    #             try:
    #                 self.solver.solve(model=self.model)
    #             except (ValueError, FloatingPointError) as err:
    #                 return False
                
    #             arr_color = self.model.colorize()
    #             self.cache[digest] = arr_color
    #         else: 
    #             # Fetch the stored array from the cache.
    #             arr_color = self.cache[digest]
    #     # end of if

    #     self.model.save_model(index=0,
    #                           fpath=fpath_model,
    #                           initializer=self.model.initializer,
    #                           params=self.model.params,
    #                           solver=self.solver,
    #                           generation=generation,
    #                           fitness=fitness)
        
    #     self.model.save_image(index=0,
    #                           fpath_morph=fpath_morph,
    #                           fpath_pattern=fpath_pattern,
    #                           arr_color=arr_color)
            
    #     return True

    # 複数island対応・save時のファイル名変更
    def save(self, 
         mode,
         dv,             
         max_generation=None,
         generation=None,
         fitness=None,
         arr_color=None,
         island=None,
         individual=None):

        dv = dv[None, :]

        self.model.initializer = self.converter.to_initializer(dv)
        self.model.params = self.converter.to_params(dv)

        if generation is None:
            str_gen = ""
        else:
            if max_generation is None:
                max_generation = 1000000
            fstr_gen = "gen-%0{}d_".format(int(np.ceil(np.log10(max_generation)))+1)
            str_gen = fstr_gen % (int(generation))

        str_isl = "" if island is None else "isl-%02d_" % island
        str_ind = "" if individual is None else "ind-%03d_" % individual

        prefix = "%s%s%s" % (str_gen, str_isl, str_ind)

        if mode == "pop":
            fpath_model = pjoin(self.dpath_population, "%smodel.json" % prefix)
            fpath_morph = pjoin(self.dpath_population, "%smorph.png" % prefix)
            fpath_pattern = pjoin(self.dpath_population, "%spattern.png" % prefix)
        elif mode == "best":
            fpath_model = pjoin(self.dpath_best, "%smodel.json" % prefix)
            fpath_morph = pjoin(self.dpath_best, "%smorph.png" % prefix)
            fpath_pattern = pjoin(self.dpath_best, "%spattern.png" % prefix)
        else:
            raise ValueError("mode should be 'pop' or 'best'")

        if arr_color is None:
            digest = get_hash_digest(dv)            
            if digest not in self.cache:                
                try:
                    self.solver.solve(model=self.model)
                except (ValueError, FloatingPointError) as err:
                    return False
                arr_color = self.model.colorize()
                self.cache[digest] = arr_color
            else: 
                arr_color = self.cache[digest]

        self.model.save_model(index=0,
                            fpath=fpath_model,
                            initializer=self.model.initializer,
                            params=self.model.params,
                            solver=self.solver,
                            generation=generation,
                            fitness=fitness)
        
        self.model.save_image(index=0,
                            fpath_morph=fpath_morph,
                            fpath_pattern=fpath_pattern,
                            arr_color=arr_color)
            
        return True
