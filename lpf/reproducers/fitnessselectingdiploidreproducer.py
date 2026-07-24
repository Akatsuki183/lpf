"""
lpf/reproducers/fitnessselectingdiploidreproducer.py

RandomTwoComponentDiploidReproducer を拡張し、
  1. n_cross 個の交叉グループを ProcessPoolExecutor で並列実行
  2. target画像に対する fitness を計算
  3. fitness上位を次世代の親候補として選択
  4. 世代履歴を溜め込まない(直前世代だけ保持)
を行うサブクラス。

前提: lpf/reproducers/randomtwocomponentdiploidreproducer.py 側の
`generate_gametes` と `cross` に @staticmethod を付与済み
(self を使っていないため、振る舞いを変えずに静的メソッド化できる)。
"""

import os
from os.path import join as pjoin
import random
import time

import numpy as np

from lpf.reproducers.randomtwocomponentdiploidreproducer import RandomTwoComponentDiploidReproducer


def _process_cross_group(male_model, female_model, n_progenies, n_gametes,
                          prob_crossover, autosomal, device,
                          solver, diploid_model_class, haploid_model_class,
                          haploid_initializer_class, alpha, beta):
    """1つの交叉グループ(親2体 → n_progenies体の子)を計算し、
    子ごとのモデルと画像を返す。ProcessPoolExecutorのワーカー内で実行される。

    self に依存しないモジュールレベル関数にしてあるので、
    ここに渡す引数だけがワーカープロセスにpickleされる
    (Reproducerインスタンス全体は送られない)。
    """

    pa_model, ma_model = RandomTwoComponentDiploidReproducer.cross(
        male_model=male_model,
        female_model=female_model,
        n_progenies=n_progenies,
        n_gametes=n_gametes,
        prob_crossover=prob_crossover,
        autosomal=autosomal,
        device=device,
    )

    model = diploid_model_class(
        paternal_model=pa_model,
        maternal_model=ma_model,
        alpha=alpha,
        beta=beta,
        device=device,
    )

    solver.solve(model=model)
    arr_color = model.colorize()

    progenies = []
    for k in range(n_progenies):
        img_ladybird, img_pattern = model.create_image(k, arr_color)

        pa_init = haploid_initializer_class(
            init_states=pa_model.initializer.init_states[None, k, :],
            init_pts=pa_model.initializer.init_pts[None, k, :, :],
        )
        progeny_pa_model = haploid_model_class(
            initializer=pa_init,
            params=pa_model.params[None, k, :],
            width=model.width, height=model.height, dx=model.dx,
            device=device,
        )

        ma_init = haploid_initializer_class(
            init_states=ma_model.initializer.init_states[None, k, :],
            init_pts=ma_model.initializer.init_pts[None, k, :, :],
        )
        progeny_ma_model = haploid_model_class(
            initializer=ma_init,
            params=ma_model.params[None, k, :],
            width=model.width, height=model.height, dx=model.dx,
            device=device,
        )

        progeny_model = diploid_model_class(
            paternal_model=progeny_pa_model,
            maternal_model=progeny_ma_model,
            alpha=alpha,
            beta=beta,
            device=device,
        )

        progenies.append({
            "MODEL": progeny_model,
            "MORPH": img_ladybird,
            "PATTERN": img_pattern,
        })

    return progenies


class FitnessSelectingDiploidReproducer(RandomTwoComponentDiploidReproducer):

    def __init__(self, *args, objectives=None, targets=None, n_procs=4, **kwargs):
        super().__init__(*args, **kwargs)

        if not objectives:
            raise ValueError("objectives must be given.")
        if not targets:
            raise ValueError("targets must be given.")

        self._objectives = objectives
        self._targets = targets
        self._n_procs = n_procs
        self._fitness_log = []  # (generation, id, fitness) の軽い履歴だけ保持

    @property
    def fitness_log(self):
        return self._fitness_log

    def evolve(self, n_generations=None, verbose=None):
        from concurrent.futures import ProcessPoolExecutor

        if not verbose:
            verbose = self._verbose
        if not n_generations:
            n_generations = self._n_generations

        n_progenies_per_cross = self._pop_size // self._n_cross

        fstr_gen = "generation-%0{}d".format(int(np.floor(np.log10(n_generations))) + 1)

        # --- 世代0(初期集団)。fitnessは未評価のまま親候補として登録するだけ。 ---
        str_gen = fstr_gen % 0
        if self._dpath_output:
            dpath_gen = pjoin(self._dpath_output, str_gen)
            os.makedirs(dpath_gen, exist_ok=True)

        prev_gen = []
        for i, model in enumerate(self._population[0]):
            str_id = "%s_model-%d" % (str_gen, i + 1)
            prev_gen.append({"ID": str_id, "MODEL": model})

            if self._dpath_output:
                fpath_model = pjoin(dpath_gen, "model_%s.json" % str_id)
                model.save_model(index=0, fpath=fpath_model)

        for i in range(1, n_generations):
            t_beg = time.time()

            str_gen = fstr_gen % i
            if self._dpath_output:
                dpath_gen = pjoin(self._dpath_output, str_gen)
                os.makedirs(dpath_gen, exist_ok=True)

            # --- 1. 親ペアを先に確定させてから、n_cross個のグループを並列実行 ---
            pairs = [(random.choice(prev_gen), random.choice(prev_gen))
                     for _ in range(self._n_cross)]

            with ProcessPoolExecutor(max_workers=self._n_procs) as executor:
                futures = [
                    executor.submit(
                        _process_cross_group,
                        male_model=male["MODEL"], female_model=female["MODEL"],
                        n_progenies=n_progenies_per_cross, n_gametes=self._n_gametes,
                        prob_crossover=self._prob_crossover, autosomal=self._autosomal,
                        device=self._device, solver=self._solver,
                        diploid_model_class=self._diploid_model_class,
                        haploid_model_class=self._haploid_model_class,
                        haploid_initializer_class=self._haploid_initializer_class,
                        alpha=self._alpha, beta=self._beta,
                    )
                    for male, female in pairs
                ]

                all_progenies = []
                for future, (male, female) in zip(futures, pairs):
                    for p in future.result():
                        p["PATERNAL"] = male["ID"]
                        p["MATERNAL"] = female["ID"]
                        all_progenies.append(p)

            # --- 2. fitness評価(target画像との比較。軽いのでメインプロセスでOK) ---
            for p in all_progenies:
                imgs = [p["MORPH"].convert("RGB")]
                p["FITNESS"] = sum(
                    float(np.sum(obj.compute(imgs, self._targets)))
                    for obj in self._objectives
                )

            # --- 3. 選択(fitnessが小さいほど良い、という前提) ---
            all_progenies.sort(key=lambda p: p["FITNESS"])
            selected = all_progenies[: self._pop_size]

            # --- 4. 保存 + 次世代の親候補を作る(履歴は溜めない) ---
            new_gen = []
            for idx, p in enumerate(selected):
                str_id = "%s_model-%d" % (str_gen, idx + 1)
                new_gen.append({"ID": str_id, "MODEL": p["MODEL"], "FITNESS": p["FITNESS"]})
                self._fitness_log.append((i, str_id, p["FITNESS"]))

                if self._dpath_output:
                    fpath_model = pjoin(dpath_gen, "model_%s.json" % str_id)
                    fpath_morph = pjoin(dpath_gen, "ladybird_%s.png" % str_id)
                    fpath_pattern = pjoin(dpath_gen, "pattern_%s.png" % str_id)

                    p["MODEL"].save_model(index=0, fpath=fpath_model)
                    p["MORPH"].save(fpath_morph)
                    p["PATTERN"].save(fpath_pattern)

            prev_gen = new_gen  # 前世代の参照はここで切れ、GCに回収される

            t_end = time.time()
            if verbose:
                print("[Generation #%d] best fitness=%.4f (%.3f sec.)"
                      % (i, selected[0]["FITNESS"], t_end - t_beg))

        return prev_gen, self._fitness_log