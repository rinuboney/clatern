(ns clatern.linear-regression-test
  (:require [clojure.test :refer :all]
            [clatern.linear-regression :refer :all]
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.linear :refer :all]))

(def epsilon 1e-5)

(defn- vec= [x y eps]
  (< (norm (sub x y)) eps))
  
(deftest test-linear-regression
  (testing "with intercepts"
    (let [X [[1 0 0]
             [0 1 0]
             [0 0 1]
             [1 1 1]]
          y [1.0 3.0 5.0 3.0]
          model (ols X y :fit-intercept true)
          coefs (:coefs model)
          intercept (:intercept model)]
      (is (= (count coefs) (dimension-count X 1)))
      (is (vec= coefs [-2.0 0.0 2.0] epsilon))
      (is (= intercept 3.0))
      (is (= (model [1 0 0]) 1.0))
      (is (= (model [0 1 0]) 3.0))
      (is (= (model [0 0 1]) 5.0))))

  (testing "without intercepts"
    (let [X [[1 0 0]
             [0 1 0]
             [0 0 1]]
          y [1.0 3.0 5.0]
          model (ols X y :fit-intercept false)
          coefs (:coefs model)
          intercept (:intercept model)]
      (is (= (count coefs) (dimension-count X 1)))
      (is (vec= coefs [1.0 3.0 5.0] epsilon))
      (is (= intercept 0.0))
      (is (= (model [1 0 0]) 1.0))
      (is (= (model [0 1 0]) 3.0))
      (is (= (model [0 0 1]) 5.0)))))
