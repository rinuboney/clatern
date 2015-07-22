(ns clatern.logistic-regression-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :refer :all]
            [clatern.logistic-regression :refer :all]))

(deftest test-logistic-regression
  (testing "logistic regression"
    (let [X [[-2 -1]
             [-1 -1]
             [-1 -2]
             [ 1  1]
             [ 1  2]
             [ 2  1]]
          y [1 1 1 2 2 2]
          model (gradient-descent X y)]
      (loop [i 0]
        (if (< i (dimension-count X 0))
          (let [v (slice X i)
                y1 (slice y i)]
            (is (= (model v) y1))
            (recur (inc i))))))))
