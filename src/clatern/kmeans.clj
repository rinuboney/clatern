(ns clatern.kmeans
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.dataset :refer :all]
            [clatern.implementations :as imp]
            [clatern.protocols :as cp]))

(def kMeans [params]
  cp/Model
  (implementation-key [m] :kmeans)

  (fit [m new-data] nil)

  (predict [m new-data] nil))

(imp/register-implementation (kMeans. {}))
