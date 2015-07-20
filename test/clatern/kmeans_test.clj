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
          clusters (kmeans X)
          centers (mapv #(get % 1) (:centroids clusters))]
      (is (= (count centers) 3))
      ; we either get: (a) one cluster at each group if
      ; each cluster is initialized to a separate point
      ; or (b) a single cluster and two empty clusters if
      ; clusters are initialized with the equal points
      (is (or (and (not= -1 (.indexOf centers v1))
                   (not= -1 (.indexOf centers v2))
                   (not= -1 (.indexOf centers v3)))
              (and (not= -1 (.indexOf centers [1/3 1/3 1/3]))
                   (not= -1 (.indexOf centers []))))))))
