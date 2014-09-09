(ns clatern.logistic-regression
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.dataset :refer :all]
            [clatern.implementations :as imp]
            [clatern.protocols :as cp]
            [clojure.core.matrix.operators :as M]
            [clatern.optimization :as optm]))

(defn sigmoid [z]
  (M// 1 (M/+ 1 (exp (M/- z)))))

(defn hypothesis [X theta]
  (sigmoid (mmul X theta)))

(defn cost-fn [h y]
  (let [m (count h)
        if-0 (mmul (transpose (log h)) y)
        if-1 (mmul (transpose (log (M/- 1 h))) (M/- 1 y))]
    (emul (/ -1 m) (M/- if-0 if-1))))

(defn grad [X y theta]
  (let [h (map #(hypothesis % theta) (rows X))
        m (count y)]    
    (M/* (/ 1 m) (mmul (transpose X) (M/- h y)))))

(defn classify [all_theta X]
  (let [all_h (for [i (keys all_theta)]
                (hypothesis X (all_theta i)))
        all_h (map vector (keys all_theta) all_h)]
    (first (last (sort-by last all_h)))))

(defrecord LogisticRegression [params]
  cp/Model
  (implementation-key [m] :logistic-regression)

  (set-options [m options]
    (assoc m :params (merge (:params m) options)))
  
  (fit [m new-data]
    (let [X (:columns (except-columns new-data ["output"]))
          X_1 (transpose (join [(repeat (column-count X) 1)] X))
          y (get-row (:columns (select-columns new-data ["output"])) 0)
          options {:alpha (:alpha (:params m))
                   :num-iters (:num-iters (:params m))}
          init-theta (vec (repeat (column-count X_1) 0))
          labels (distinct y)
          all_y (for [i labels] (map #(if (= i %) 1 0) y))
          all_theta (map #(optm/gradient-descent X_1 % grad init-theta options) all_y)          
          all_theta (zipmap labels all_theta)]
      (assoc m :params (assoc (:params m) :all_theta all_theta :labels labels))))

  (predict [m new-data]
    (let [X (:columns new-data)
          X_1 (transpose (join [(repeat (column-count X) 1)] X)) 
          all_theta (:all_theta (:params m))
          cnames (conj (:column-names new-data) "output")
          output (map #(classify all_theta %) (rows X_1))]
      (dataset cnames (map conj (rows (transpose X)) output)))))

(imp/register-implementation (LogisticRegression. {}))
