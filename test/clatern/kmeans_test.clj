(ns clatern.kmeans-test
  (:require [clojure.test :refer :all]
            [clatern.kmeans :refer :all]
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :as M]))

(deftest test-three-clusters
  (testing "three clusters"
    (let [v1 [0 0 1]
          v2 [0 1 0]
          v3 [1 0 0]
          X (into (take 3 (repeat v1))
                    (into (take 3 (repeat v2))
                    (take 3 (repeat v3))))
          clusters (kmeans X :num-iters 10)
          centers (mapv #(get % 1) (:centroids clusters))]
      (is (= (count centers) 3))
      (is (and (not= -1 (.indexOf centers v1))
               (not= -1 (.indexOf centers v2))
               (not= -1 (.indexOf centers v3)))))))

