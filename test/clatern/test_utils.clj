(ns clatern.test-utils
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.linear :refer [norm]]))

(def eps 1e-5)

(defn float-equals
  "Compare floats within epsilon"
  [x y eps]
  (< (Math/abs (- x y)) eps))

(defn vector-equals
  "Compare vectors or matrices of floats within epsilon"
  [v1 v2 eps]
  (< (norm (sub v1 v2)) eps))
