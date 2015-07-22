(ns clatern.linear-regression-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :refer :all]
            [clatern.test-utils :refer :all]
            [clatern.linear-regression :refer :all]))

(deftest test-linear-regression
  (testing "linear regression on vector"
    (let [X [[1 0 0]
             [0 1 0]
             [0 0 1]
             [1 1 1]]
          y [1 3 5 7]
          model (ols X y)]
      ; test predictions on individual vectors
      (loop [i 0]
        (if (< i (dimension-count X 0))
          (let [v (slice X i)
                y1 (slice y i)]
                (is (float-equals (model v) y1 eps))
                (recur (inc i)))))
      ; test predictions on multiple vectors
      (is (vector-equals (model X) y eps)))))
