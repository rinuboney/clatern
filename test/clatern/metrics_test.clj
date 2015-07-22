(ns clatern.metrics-test
  (:require [clojure.test :refer :all]
            [clatern.metrics :refer :all]))

(def eps 1e-5)

(defn- float-equals
  "Compare floats within epsilon"
  [x y eps]
  (< (Math/abs (- x y)) eps))

(deftest test-equal-vectors
  (testing "Equal vectors"
    (let [v [0 1]
          expected 0.0]
      (is (float-equals (euclidean-distance v v) expected eps))
      (is (float-equals (squared-euclidean-distance v v) expected eps))
      (is (float-equals (cosine-distance v v) expected eps))
      (is (float-equals (manhattan-distance v v) expected eps)))))

(deftest test-orthog-vectors
  (testing "Orthogonal vectors"
    (let [v1 [0 1]
          v2 [1 0]]
      (is (float-equals (euclidean-distance v1 v2) (Math/sqrt 2.0) eps))
      (is (float-equals (squared-euclidean-distance v1 v2) 2.0 eps))
      (is (float-equals (cosine-distance v1 v2) 1.0 eps))
      (is (float-equals (manhattan-distance v1 v2) 2.0 eps)))))

(deftest test-diff-vectors
  (testing "Different vectors"
    (let [v1 [0.75 0.3]
          v2 [0.25 0.5]]
      (is (float-equals (euclidean-distance v1 v2) 0.5385164807 eps))
      (is (float-equals (squared-euclidean-distance v1 v2) 0.29 eps))
      (is (float-equals (manhattan-distance v1 v2) 0.7 eps))
      (is (float-equals (cosine-distance v1 v2) 0.2525906813163402 eps)))))
