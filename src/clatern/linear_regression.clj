(ns clatern.linear-regression
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.dataset :refer :all]
            [clojure.core.matrix.linear :refer :all]))

;; Ordinary Least Squares
;; ======================
;;  X : input data
;;  y : target data

(defn- center-data [X y fit-intercept]
  (let [n-samples (dimension-count X 0)
        n-features (dimension-count X 1)]
    (if fit-intercept
      (let [X-mean (emul (reduce add (rows X)) (/ 1.0 n-features))
            X-centered (sub X X-mean)
            y-mean (emul (esum y) (/ 1.0 n-samples))
            y-centered (sub y y-mean)]
        [X-centered y-centered X-mean y-mean])
      (let [X-mean (zero-vector n-samples)
            y-mean 0]
        [X y X-mean y-mean]))))

(defn- compute-intercept [X-mean y-mean coefs fit-intercept]
  (if fit-intercept
    (sub y-mean (dot X-mean (transpose coefs)))
    0.0))
        
(defn- predict [coefs intercept v]
  (+ (mmul coefs v) intercept))

(defrecord LinearRegression [coefs intercept]
  clojure.lang.IFn
  (invoke [this v] (predict coefs intercept v))
  (applyTo [this args] (clojure.lang.AFn/applyToHelper this args)))

(defn ols [X y & {:keys [fit-intercept]
                  :or {fit-intercept false}}]
  {:pre [(let [X-rows (dimension-count X 0)
               X-columns (dimension-count X 1)]
           (if fit-intercept
             (> X-rows X-columns)
             (>= X-rows X-columns)))]}
  (let [[X-centered y-centered X-mean y-mean] (center-data X y fit-intercept)
        Xt (transpose X-centered)
        Xt-X (mmul Xt X-centered)
        coefs (matrix (mmul (inverse Xt-X) Xt y-centered))
        intercept (compute-intercept X-mean y-mean coefs fit-intercept)]
    (LinearRegression. coefs intercept)))


;; TODO: implement gradient descent support 
