(ns clatern.naive-bayes-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :refer :all]
            [clatern.naive-bayes :refer :all]))

(deftest test-gaussian-naive-bayes
  (testing "gaussian naive bayes"
    (let [X [[-2 -1]
             [-1 -1]
             [-1 -2]
             [ 1  1]
             [ 1  2]
             [ 2  1]]
          y [1 1 1 2 2 2]
          model (gaussian-nb X y)]
      (is (= (:priors {1 (/ 3.0 6.0), 
                       2 (/ 3.0 6.0)})))
      (loop [i 0]
        (if (< i (dimension-count X 0))
          (let [v (slice X i)
                y1 (slice y i)]
            (is (= (model v) y1))
            (recur (inc i))))))))
