(ns clatern.knn-test
  (:require [clojure.test :refer :all]
            [clatern.knn :refer :all]))

(deftest test-knn
  (testing "k=1"
    (let [X [[0 0 0 1]
             [0 0 1 0]
             [0 1 0 0]
             [1 0 0 0]]
          y [0 1 0 1]
          v1 [0 0 0 0.95]
          y1 (knn X y v1 :k 1)
          expected 0]
      (is (= y1 expected))))

  (testing "k=3"
    (let [X [[0 0 0 1]
             [0 0 1 0]
             [0 1 0 0]
             [1 0 0 0]]
          y [0 1 0 1]
          v1 [0 0.95 0.95 0.95]
          y1 (knn X y v1 :k 3)
          expected 0]
      (is (= y1 expected)))))

