(ns clatern.decision-tree-test
  (:require [clojure.test :refer :all]
            [clatern.decision-tree :refer :all]
            [clatern.protocols :refer [predict-prob predict-log-prob]]
            [clatern.test-utils :refer :all]))

(deftest test-gini
  (testing "equal thirds"
    (let [y [0 0 0 1 1 1 2 2 2]
          indices (range (count y))
          expected (/ 2.0 3.0)
          observed (#'clatern.decision-tree/gini-impurity y indices)]
      (is (float-equals expected observed eps))))

  (testing "all the same class"
    (let [y (vector (repeat 9 0))
          indices (range (count y))
          expected 0.0
          observed (#'clatern.decision-tree/gini-impurity y indices)]
      (is (float-equals expected observed eps))))

  (testing "empty set"
    (let [y (vector (repeat 9 0))
          indices []
          expected 1.0
          observed (#'clatern.decision-tree/gini-impurity y indices)]
      (is (float-equals expected observed eps)))))


(deftest test-find-optimal-threshold
  (testing "two classes"
    (let [X [[0]
             [1]
             [2]
             [3]
             [4]
             [5]]
          y [0 0 0 1 1 1]
          indices (range (count y))
          [t cost left right] (#'clatern.decision-tree/find-optimal-threshold X y indices 0)
          expected-t 2
          expected-cost 0.0]
      (is (float-equals expected-t t eps))
      (is (float-equals expected-cost cost eps))))

  (testing "single class"
    (let [X [[0]
             [1]
             [2]
             [3]
             [4]
             [5]]
          y [0 0 0 0 0 0]
          indices (range (count y))
          [t cost left right] (#'clatern.decision-tree/find-optimal-threshold X y indices 0)
          expected-cost 0.0]
      (is (float-equals expected-cost cost eps)))))


(deftest test-find-best-split
  (testing "two classes, three features"
    (let [X [[0 0 5]
             [1 5 1]
             [2 1 4]
             [3 4 2]
             [4 2 3]
             [5 3 1]]
          y [0 0 0 1 1 1]
          indices (range (count y))
          [feature-idx threshold cost left right] (#'clatern.decision-tree/find-best-split
                                                   X y range indices)
          expected-cost 0.0
          expected-idx 0
          expected-threshold 2]
      (is (= expected-idx feature-idx))
      (is (float-equals expected-cost cost eps))
      (is (float-equals expected-threshold threshold eps)))))


(deftest test-split-node
  (testing "perfect split"
    (let [y [0 0 0 1 1 1]
          max-depth 3
          left [0 1 2]
          right [3 4 5]
          depth 2]
      (is (= (#'clatern.decision-tree/split-node? y max-depth depth left right) true))))

  (testing "empty left"
    (let [y [0 0 0 1 1 1]
          max-depth 3
          left []
          right [0 1 2 3 4 5]
          depth 2]
      (is (= (#'clatern.decision-tree/split-node? y max-depth depth left right) false))))

  (testing "empty right"
    (let [y [0 0 0 1 1 1]
          max-depth 3
          left [0 1 2 3 4 5]
          right []
          depth 2]
      (is (= (#'clatern.decision-tree/split-node? y max-depth depth left right) false))))

  (testing "exceeded max depth"
    (let [y [0 0 0 1 1 1]
          max-depth 3
          left [0 1 2]
          right [3 4 5]
          depth 4]
      (is (= (#'clatern.decision-tree/split-node? y max-depth depth left right) false))))

  (testing "homogenous labels"
    (let [y [0 0 0 0 0 0]
          max-depth 3
          left [0 1 2]
          right [3 4 5]
          depth 2]
      (is (= (#'clatern.decision-tree/split-node? y max-depth depth left right) false)))))

(deftest test-train-tree
  (testing "2 classes, 1 feature, 2 cluster"
    (let [X [[2]
             [2]
             [4]
             [4]]
          y [0 0 1 1]
          max-depth 3
          tree (#'clatern.decision-tree/train-tree X y max-depth range (range (count y)))]
      (is (= (get-in tree [:left :type] :leaf)))
      (is (= (get-in tree [:right :type] :leaf)))
      (is (= (get-in tree [:left :label-counts]) {0 2, 1 0}))
      (is (= (get-in tree [:right :label-counts]) {0 0, 1 2}))
      (is (= (get-in tree [:threshold]) 2))))

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
          max-depth 3
          tree (#'clatern.decision-tree/train-tree X y max-depth range (range (count y)))]
      (is (= (get-in tree [:left :left :type] :leaf)))
      (is (= (get-in tree [:left :right :type] :leaf)))
      (is (= (get-in tree [:right :left :type] :leaf)))
      (is (= (get-in tree [:right :right :type] :leaf)))
      (is (= (get-in tree [:left :left :label-counts]) {0 2, 1 0}))
      (is (= (get-in tree [:left :right :label-counts]) {0 0, 1 2}))
      (is (= (get-in tree [:right :left :label-counts]) {0 0, 1 2}))
      (is (= (get-in tree [:right :right :label-counts]) {0 2, 1 0}))
      (is (= (get-in tree [:threshold]) 0))
      (is (= (get-in tree [:left :threshold]) 0))
      (is (= (get-in tree [:right :treshold] 0))))))

(deftest test-decision-tree
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
          max-depth 3
          dt (decision-tree X y)
          prob-v1 (predict-prob dt v1)
          prob-v2 (predict-prob dt v2)
          log-prob-v1 (predict-log-prob dt v1)
          log-prob-v2 (predict-log-prob dt v2)]
      (is (some? (:tree dt)))
      (is (= (dt v1) 0))
      (is (= (dt v2) 1))
      (is (= (dt v3) 1))
      (is (= (dt v4) 0))
      (is (and 
           (float-equals (prob-v1 0) 1.0 eps)
           (float-equals (prob-v1 1) 0.0 eps)))
      (is (and 
           (float-equals (prob-v2 1) 1.0 eps)
           (float-equals (prob-v2 0) 0.0 eps)))
      (is (and 
           (float-equals (log-prob-v1 0) 0.0 eps)
           (< (log-prob-v1 1) (java.lang.Math/log eps))))
      (is (and 
           (float-equals (log-prob-v2 1) 0.0 eps)
           (< (log-prob-v2 0) (java.lang.Math/log eps))))))

  (testing "2 classes, 2 features, 4 clusters with elements specified"
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
          max-depth 3
          dt (decision-tree X y :elems [0 1 2 3])
          prob-v1 (predict-prob dt v1)
          prob-v2 (predict-prob dt v2)]
      (is (some? (:tree dt)))
      (is (= (dt v1) 0))
      (is (= (dt v2) 1))
      (is (= (dt v3) 0))
      (is (= (dt v4) 1))
      (is (float-equals (prob-v1 0) 1.0 eps))
      (is (float-equals (prob-v2 1) 1.0 eps)))))
      
