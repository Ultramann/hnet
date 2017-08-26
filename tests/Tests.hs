module Main where

import Neural.Matrix
import Numeric.LinearAlgebra
import Test.Hspec

{- Simple Net
   1                    ----     ----      ----       w2           b2       zL       aL
      [3, 2, -2] -> 1  |-1.5| = |-0.5| -> |  0 |    -----         ----      ---      ---
   2    --w1--          -b1-     -z1-      -a1-   [-2, 0.5] -> 7 | -3 | -> | 4 | -> | 4 |
      [-1, 2, 3] -> 12 | +2 | = | 14 | -> | 14 |    -----         ----      ---      ---
   3                    ----     ----      ----
-}

zeroVec2 = vector [0,0]
zeroVec3 = vector [0,0,0]
zeroD1 = Dense (matrix 3 (replicate 6 0)) zeroVec3
zeroNet = Network [zeroD1, Dense (matrix 1 [0,0,0]) (vector [0])] [relu, relu]
nonZeroVec2 = vector [-1.5,2]
nonZeroVec3 = vector [1,2,3]
simpleD1 = Dense (matrix 2 [3,-1,2,2,-2,3]) nonZeroVec2
w2 = matrix 1 [-2,0.5]
b2 = vector [-3]
simpleNet = Network [simpleD1, Dense w2 b2] [relu, relu]

main :: IO ()
main = hspec $ do
  describe "Check dense:" $ do

    describe "feed forward:" $ do
      it "passes zero through zero case" $ do
        feedForward zeroD1 zeroVec2 `shouldBe` zeroVec3
      it "passes non-zero through zero case" $ do
        feedForward zeroD1 nonZeroVec2 `shouldBe` zeroVec3
      it "passes zero through simple case" $ do
        feedForward simpleD1 zeroVec3 `shouldBe` nonZeroVec2
      it "passes simple through simple case" $ do
        feedForward simpleD1 nonZeroVec3 `shouldBe` vector [-0.5,14]

    describe "pass backward:" $ do
      it "passes zero through zero case" $ do
        passBackward zeroD1 zeroVec3 `shouldBe` zeroVec2
      it "passes non-zero through zero case" $ do
        passBackward zeroD1 nonZeroVec3 `shouldBe` zeroVec2
      it "passes zero through simple case" $ do
        passBackward simpleD1 zeroVec2 `shouldBe` zeroVec3
      it "passes simple through simple case" $ do
        passBackward simpleD1 nonZeroVec2 `shouldBe` vector [-6.5,1,9]

  describe "Check relu:" $ do

    describe "feed forward:" $ do
      it "passes zero case" $ do
        feedForward relu zeroVec2 `shouldBe` zeroVec2
      it "passes positives case" $ do
        feedForward relu nonZeroVec3 `shouldBe` nonZeroVec3
      it "passes negatives case" $ do
        feedForward relu (negate nonZeroVec3) `shouldBe` zeroVec3
      it "passes positives & negatives case" $ do
        feedForward relu nonZeroVec2 `shouldBe` vector [0,2]

    describe "pass backward:" $ do
      it "passes zero case" $ do
        passBackward relu zeroVec2 `shouldBe` zeroVec2
      it "passes positives case" $ do
        passBackward relu nonZeroVec3 `shouldBe` vector [1,1,1]
      it "passes negatives case" $ do
        passBackward relu (negate nonZeroVec3) `shouldBe` zeroVec3
      it "passes positives & negatives case" $ do
        passBackward relu nonZeroVec2 `shouldBe` vector [0,1]
{-
  describe "Check forward pass" $ do
    it "passes trivial case" $ do
      let zp = [(zeroVec, zeroVec), (zeroVec, zeroVec),
                (vector [0.0], vector [0.0])]
      (forwardPass zeroVec zeroNet) `shouldBe` zp
    it "passes simple case" $ do
      let fp = [(vector [1.0,2.0,3.0], vector [1.0,2.0,3.0]),
                (vector [-0.5,14.0], vector [0.0,14.0]),
                (vector [4.0], vector [4.0])]
      (forwardPass vTest simpleNet) `shouldBe` fp

  describe "Check chain" $ do
    it "passes trivial case" $ do
      let (_, _, zm) = last zeroNet
      let ze = vector [0.0]
      (chain (zeroVec, zm) ze) `shouldBe` zeroVec
    it "passes simple case" $ do
      let tl = vector [-0.5,14]
      let (_, _, lw) = last simpleNet
      let e = vector [0.5]
      let chained = vector [0,0.25]
      (chain (tl, lw) e) `shouldBe` chained

  describe "Check back prop" $ do
    it "passes simple case" $ do
      let tls = [zeroVec, vector [0]]
      let lws = map (\(_, _, x) -> x) zeroNet
      (backProp (vector [0]) tls lws) `shouldBe` tls
-}
