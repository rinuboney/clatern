(ns clatern.linear-regression
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.dataset :refer :all]
            [clatern.protocols :as cp]
            [clatern.implementations :as imp]))

(defn- ols [MX y]
  (let [X (transpose (join [(repeat (column-count MX) 1)] MX))
        Xt (matrix (transpose X))
        Xt-X (mmul Xt X)]
    (mmul (inverse Xt-X) Xt y)))

(defn- ols-predict [coefs X]
  {:pre [(= (count coefs) (+ 1 (count X)))]}
  (let [X_1 (conj X 1)
        product (map * X_1 coefs)]
    (reduce + product)))

(defrecord LinearRegression [params]
  cp/Model
  (implementation-key [m] :linear-regression)
  
  (fit [m new-data]
    (let [X (:columns (except-columns new-data ["output"]))
          y (get-row (:columns (select-columns new-data ["output"])) 0)]
      (assoc m :params (assoc (:params m) :coefs (matrix (ols X y))))))
  
  (predict [m new-data]
    (let [X (:columns new-data)
          coefs (:coefs (:params m))
          cnames (conj (:column-names new-data) "output")]
      (dataset cnames (map  #(conj % (ols-predict coefs %))
                            (rows (transpose X)))))))

(imp/register-implementation (LinearRegression. {}))
