(ns clatern.random-forest-test
  (:require [clojure.test :refer :all]
            [clatern.random-forest :refer :all]
            [clatern.protocols :refer [predict-prob predict-log-prob]]
            [clatern.test-utils :refer :all]))

(deftest test-sampling-tools
  (testing "sqrt-round"
    (is (= (#'clatern.random-forest/sqrt-round 1) 1))
    (is (= (#'clatern.random-forest/sqrt-round 2) 1))
    (is (= (#'clatern.random-forest/sqrt-round 3) 2))
    (is (= (#'clatern.random-forest/sqrt-round 4) 2))))

(deftest test-train-forest
  (testing "2 classes, 2 features, 4 clusters"
    (let [X [[0 0]
             [0 0]
             [0 1]
             [0 1]
             [1 0]
             [1 0]
             [1 1]
             [1 1]]
          y [0 0 1 1 1 1 0 0]
          v1 [0 0]
          v2 [0 1]
          v3 [1 0]
          v4 [1 1]
          n-trees 10
          max-depth 5
          trees (#'clatern.random-forest/train-forest X y n-trees max-depth)]
      (is (= (count trees) n-trees)))))

(deftest test-random-forest
  (testing "2 classes, 2 features, 4 clusters"
    (let [X [[0 0 0]
             [0 0 0]
             [0 1 0]
             [0 1 0]
             [1 0 1]
             [1 0 1]
             [1 1 1]
             [1 1 1]]
          y [0 0 0 0 1 1 1 1]
          v1 [0 0 0]
          v2 [0 1 0]
          v3 [1 0 1]
          v4 [1 1 1]
          rf (random-forest X y :max-depth 1)]
      (is (some? (:trees rf)))
      (is (= (rf v1) 0))
      (is (= (rf v2) 0))
      (is (= (rf v3) 1))
      (is (= (rf v4) 1)))))
