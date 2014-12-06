(ns clatern.logistic-regression
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.dataset :refer :all]
            [clojure.core.matrix.operators :as M]
            [clatern.optimization :as optm]))

(defn- sigmoid [z]
  (M// 1 (M/+ 1 (exp (M/- z)))))

(defn- hypothesis [X theta]
  (sigmoid (mmul X theta)))

(defn- cost-fn [h y]
  (let [m (count h)
        if-0 (mmul (transpose (log h)) y)
        if-1 (mmul (transpose (log (M/- 1 h))) (M/- 1 y))]
    (emul (/ -1 m) (M/- if-0 if-1))))

(defn- grad [X y theta lambda]
  (let [h (map #(hypothesis % theta) (rows X))
        m (count y)
        theta0 (vec (conj (rest theta) 0))]    
    (M/+ (M/* (/ 1 m) (mmul (transpose X) (M/- h y)))
         (M/* lambda theta0))))

(defn- classify [all_theta v]
  (let [v_1 (cons 1 v)
        all_h (for [i (keys all_theta)]
                (hypothesis v_1 (all_theta i)))
        all_h (map vector (keys all_theta) all_h)]
    (first (last (sort-by last all_h)))))

(defrecord LogisticRegression [thetas]
  clojure.lang.IFn
  (invoke [this v] (classify thetas v))
  (applyTo [this args] (clojure.lang.AFn/applyToHelper this args)))

(defn gradient-descent [X y & {:keys [alpha lambda num-iters]
                               :or {alpha 0.1
                                    lambda 1
                                    num-iters 100}}]
  (let [X_1 (join-along 1 (transpose [(repeat (row-count X) 1)]) X)
        labels (distinct y)
        init-theta (vec (repeat (column-count X_1) 0))
        all_y (for [i labels] (map #(if (= i %) 1 0) y))
        all_theta (map #(optm/gradient-descent X_1 % grad init-theta alpha lambda num-iters) all_y)          
        all_theta (zipmap labels all_theta)]
    (LogisticRegression. all_theta)))
