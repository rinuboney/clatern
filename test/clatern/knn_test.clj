(ns clatern.knn-test
  (:require [clojure.test :refer :all]
            [clatern.knn :refer :all]
            [clatern.metrics :refer :all]))

(deftest test-knn
  (testing "k=1"
    (let [X [[0 0 0 1]
             [0 0 1 0]
             [0 1 0 0]
             [1 0 0 0]]
          y [0 1 0 1]
          v [0 0 0 0.95]
          k 1
          model (knn X y :k k)
          y1 (model v)
          expected 0]
      (is (= (:k model) k))
      (is (= y1 expected))))

  (testing "k=3"
    (let [X [[0 0 0 1]
             [0 0 1 0]
             [0 1 0 0]
             [1 0 0 0]]
          y [0 1 0 1]
          v [0 0.95 0.95 0.95]
          k 3
          model (knn X y :k k)
          y1 (model v)
          expected 0]
      (is (= (:k model) k))
      (is (= y1 expected))))

  (testing "k=3 cosine"
    (let [X [[0 0 0 1]
             [0 0 1 0]
             [0 1 0 0]
             [1 0 0 0]]
          y [0 1 0 1]
          v [0 0.95 0.95 0.95]
          k 3
          model (knn X y :k k :distance cosine-distance)
          y1 (model v)
          expected 0]
      (is (= (:k model) k))
      (is (= y1 expected))))

  (testing "k=3 manhattan"
    (let [X [[0 0 0 1]
             [0 0 1 0]
             [0 1 0 0]
             [1 0 0 0]]
          y [0 1 0 1]
          v [0 0.95 0.95 0.95]
          k 3
          model (knn X y :k k :distance manhattan-distance)
          y1 (model v)
          expected 0]
      (is (= (:k model) k))
      (is (= y1 expected)))))

