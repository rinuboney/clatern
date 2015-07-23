(ns clatern.linear-regression
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.dataset :refer :all]))

;; Ordinary Least Squares
;; ======================
;;  X : input data
;;  y : target data

(defn- predict [coefs v]
  {:pre [(let [ncoefs (dimension-count coefs 0)
               lastdim (- (dimensionality v) 1)]
           (or (= lastdim 0)
               (= lastdim 1))
           (= ncoefs (+ 1 (dimension-count v lastdim))))]}
  (let [ndim (dimensionality v)
        v_1 (if (= ndim 1)
              (join v 1)
              (join-along 1 v (repeat (row-count v) 1)))]
    (mmul v_1 coefs)))

(defrecord LinearRegression [coefs]
  clojure.lang.IFn
  (invoke [this v] (predict coefs v))
  (applyTo [this args] (clojure.lang.AFn/applyToHelper this args)))

(defn ols [X y]
  (let [X_1 (join-along 1 X (transpose [(repeat (row-count X) 1)]))
        Xt (transpose X_1)
        Xt-X (mmul Xt X_1)
        coefs (matrix (mmul (inverse Xt-X) Xt y))]
    (LinearRegression. coefs)))


;; TODO: implement gradient descent support 
