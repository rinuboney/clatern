(ns clatern.linear-regression
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.dataset :refer :all]))

(defn- predict [coefs v]
  {:pre [(= (count coefs) (+ 1 (count v)))]}
  (let [v_1 (conj v 1)
        product (map * v_1 coefs)]
    (reduce + product)))

(defrecord LinearRegression [coefs]
  clojure.lang.IFn
  (invoke [this v] (predict coefs v))
  (applyTo [this args] (clojure.lang.AFn/applyToHelper this args)))

(defn ols [X y]
  (let [X_1 (join-along 1 (transpose [(repeat (row-count X) 1)]) X)
        Xt (transpose X_1)
        Xt-X (mmul Xt X_1)
        coefs (matrix (mmul (inverse Xt-X) Xt y))]
    (LinearRegression. coefs)))
