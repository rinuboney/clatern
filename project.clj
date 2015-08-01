(defproject clatern "0.1.0-SNAPSHOT"
  :description "Machine Learning in Clojure"
  :url "https://github.com/rinuboney/clatern"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.6.0"]
                 [net.mikera/core.matrix "0.29.1"]
                 [org.clojure/data.csv "0.1.2"]
                 [net.mikera/core.matrix.stats "0.4.0"]]
  :scm {:name "git"
        :url "https://github.com/rinuboney/clatern"}
  :plugins [[codox "0.8.13"]]
  :codox {:src-dir-uri "https://github.com/rinuboney/clatern/blob/master/"
          :src-linenum-anchor-prefix "L"
          :defaults {:doc/format :markdown}})
