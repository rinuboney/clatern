(ns clatern.kmeans-test
  (:require [clojure.test :refer :all]
            [clatern.kmeans :refer :all]
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :as M]
            [clatern.metrics :refer :all]))

(def v1 [0 0 1])
(def v2 [0 1 0])
(def v3 [1 0 0])
(def X [v1 v1 v1 v2 v2 v2 v3 v3 v3])

(deftest test-three-clusters
  (testing "three clusters"
    (let [clusters (kmeans X :num-iters 10)
          centers (mapv second (:centroids clusters))]
      (is (= (count centers) 3))
      (is (and (not= -1 (.indexOf centers v1))
               (not= -1 (.indexOf centers v2))
               (not= -1 (.indexOf centers v3))))))

  (testing "three clusters with cosine distance"
    (let [clusters (kmeans X :num-iters 10 :dist-metric cosine-distance)
          centers (mapv second (:centroids clusters))]
      (is (= (count centers) 3))
      (is (and (not= -1 (.indexOf centers v1))
               (not= -1 (.indexOf centers v2))
               (not= -1 (.indexOf centers v3))))))

  (testing "three clusters with manhattan distance"
    (let [clusters (kmeans X :num-iters 10 :dist-metric manhattan-distance)
          centers (mapv second (:centroids clusters))]
      (is (= (count centers) 3))
      (is (and (not= -1 (.indexOf centers v1))
               (not= -1 (.indexOf centers v2))
               (not= -1 (.indexOf centers v3)))))))

