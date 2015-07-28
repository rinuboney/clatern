(ns clatern.preprocessing-test
  (:require [clojure.test :refer :all]
            [clatern.preprocessing :refer :all]
            [clatern.test-utils :refer :all]
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.stats :refer :all]))

; generated from a Gaussian distribution with
; mean 2 and std 5
(def M [[-10.02092749940913 -3.0541147087529392 -7.4236835305502105]
             [6.48294814891511 1.6894502032619167 3.68696333243396]
             [4.53753510291792 5.132944877019162 5.16921635620113]
             [4.8842273991086405 -2.3110141708928467 1.4164146024978965]
             [1.7312366437838975 -1.622939509273678 7.294344192024754]
             [4.306326075575135 6.197662263980167 0.7941109924023133]
             [3.1841522987173296 10.166188800073252 5.224981038474112]
             [7.072204138729375 -1.9441100401058193 9.935243317372386]
             [9.078906600837602 7.97115739685287 -1.7599462004500532]
             [1.4615009695396037 -3.282974506518837 0.1094944970660856]])

(deftest test-mean-normalizer
  (testing "mean-normalizer on vector"
    (let [V [(mean (get-column M 0))
             (mean (get-column M 1))
             (mean (get-column M 2))]
          normalizer (mean-normalizer V)
          U (normalizer V)]

      ; verify that input data are not centered and scaled
      (is (not (float-equals (mean V) 0.0 eps)))
      (is (not (float-equals (sd V) 1.0 eps)))

      ; verify that outputs are centered and scaled
      (is (= (shape U) (shape V)))
      (is (float-equals (mean U) 0.0 eps))
      (is (float-equals (sd U) 1.0 eps))))
  
  (testing "mean-normalizer trained on a matrix, applied to a matrix and vector"
    ; choose V so that values will be shifted to [0 0 0]
    (let [V [(mean (get-column M 0))
             (mean (get-column M 1))
             (mean (get-column M 2))]
          normalizer (mean-normalizer M)
          N (normalizer M)
          U (normalizer V)]

      ; verify that input data are not centered and scaled
      (is (not (vector-equals (mean M) [0.0 0.0 0.0] eps)))
      (is (not (vector-equals (sd M) [1.0 1.0 1.0] eps)))
      (is (not (float-equals (mean V) 0.0 eps)))
      (is (not (float-equals (sd V) 1.0 eps)))
      
      ; verify that outputs are centered and scaled
      (is (= (shape N) (shape M)))
      (is (= (shape V) (shape U)))
      (is (vector-equals (mean N) [0.0 0.0 0.0] eps))
      (is (vector-equals (sd N) [1.0 1.0 1.0] eps))
      (is (vector-equals (mean U) [0.0 0.0 0.0] eps)))))


(deftest test-min-max-scaler
  (testing "min-max-scaler trained on a matrix, applied to vector and matrix"
    (let [V [-10.02092749940913 10.166188800073252 9.935243317372386]
          scaler (min-max-scaler M)
          N (scaler M)
          U (scaler V)]
    
      ; verify that input data are not scaled
      (is (not (float-equals (emin V) 0.0 eps)))
      (is (not (float-equals (emax V) 1.0 eps)))
      (is (not (float-equals (emin M) 0.0 eps)))
      (is (not (float-equals (emax M) 1.0 eps)))
      
      ; verify that output data are scaled
      (is (= (shape N) (shape M)))
      (is (= (shape V) (shape U)))
      (is (float-equals (emin U) 0.0 eps))
      (is (float-equals (emax U) 1.0 eps))
      (is (float-equals (emin N) 0.0 eps))
      (is (float-equals (emax N) 1.0 eps)))))


