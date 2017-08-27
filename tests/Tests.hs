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

zeroVec1 = vector [0]
zeroVec2 = vector [0,0]
zeroVec3 = vector [0,0,0]
zeroD1 = Dense (matrix 3 (replicate 6 0)) zeroVec3
zeroNet = Network [zeroD1, Dense (matrix 1 [0,0,0]) zeroVec1] [relu, relu]
nonZeroVec1 = vector [1]
nonZeroVec2 = vector [-1.5,2]
nonZeroVec3 = vector [1,2,3]
simpleD1 = Dense (matrix 2 [3,-1,2,2,-2,3]) nonZeroVec2
w2 = matrix 1 [-2,0.5]
b2 = vector [-3]
simpleNet = Network [simpleD1, Dense w2 b2] [relu, relu]

main :: IO ()
main = hspec $ do
-------------------------------------------------------------------------------------------------

-------------------------------------------------------------------------------------------------
  describe "Test feedForward:" $ do
    describe "- dense:" $ do
      it "passes zero through zero case" $ do
        feedForward zeroD1 zeroVec2 `shouldBe` zeroVec3
      it "passes non-zero through zero case" $ do
        feedForward zeroD1 nonZeroVec2 `shouldBe` zeroVec3
      it "passes zero through simple case" $ do
        feedForward simpleD1 zeroVec3 `shouldBe` nonZeroVec2
      it "passes simple through simple case" $ do
        feedForward simpleD1 nonZeroVec3 `shouldBe` vector [-0.5,14]
    describe "- relu:" $ do
      it "passes zero case" $ do
        feedForward relu zeroVec2 `shouldBe` zeroVec2
      it "passes positives case" $ do
        feedForward relu nonZeroVec3 `shouldBe` nonZeroVec3
      it "passes negatives case" $ do
        feedForward relu (negate nonZeroVec3) `shouldBe` zeroVec3
      it "passes positives & negatives case" $ do
        feedForward relu nonZeroVec2 `shouldBe` vector [0,2]
-------------------------------------------------------------------------------------------------

-------------------------------------------------------------------------------------------------
  describe "Test passBackward:" $ do
    describe "- dense:" $ do
      it "passes zero through zero case" $ do
        passBackward zeroD1 zeroVec3 `shouldBe` zeroVec2
      it "passes non-zero through zero case" $ do
        passBackward zeroD1 nonZeroVec3 `shouldBe` zeroVec2
      it "passes zero through simple case" $ do
        passBackward simpleD1 zeroVec2 `shouldBe` zeroVec3
      it "passes non-zero through simple case" $ do
        passBackward simpleD1 nonZeroVec2 `shouldBe` vector [-6.5,1,9]
    describe "- relu:" $ do
      it "passes zero case" $ do
        passBackward relu zeroVec2 `shouldBe` zeroVec2
      it "passes positives case" $ do
        passBackward relu nonZeroVec3 `shouldBe` vector [1,1,1]
      it "passes negatives case" $ do
        passBackward relu (negate nonZeroVec3) `shouldBe` zeroVec3
      it "passes positives & negatives case" $ do
        passBackward relu nonZeroVec2 `shouldBe` vector [0,1]
-------------------------------------------------------------------------------------------------

-------------------------------------------------------------------------------------------------
  describe "Test forwardPass:" $ do
    it "passes zero through zero case" $ do
      forwardPass zeroVec2 zeroNet `shouldBe` ([zeroVec2, zeroVec3, zeroVec1],
                                               [zeroVec2, zeroVec3, zeroVec1])
    it "passes non-zero through zero case" $ do
      forwardPass nonZeroVec2 zeroNet `shouldBe` ([nonZeroVec2, zeroVec3, zeroVec1],
                                                  [nonZeroVec2, zeroVec3, zeroVec1])
    it "passes zero through simple case" $ do
      forwardPass zeroVec3 simpleNet `shouldBe` ([zeroVec3, nonZeroVec2, vector [-2]],
                                                 [zeroVec3, vector [0,2], zeroVec1])
    it "passes non-zero through simple case" $ do
      forwardPass nonZeroVec3 simpleNet `shouldBe` ([nonZeroVec3, vector [-0.5,14], vector [4]],
                                                    [nonZeroVec3, vector [0,14], vector [4]])
-------------------------------------------------------------------------------------------------

-------------------------------------------------------------------------------------------------
  describe "Test backPropagate:" $ do
    it "passes zero through zero case" $ do
      backPropagate zeroVec1 zeroNet [zeroVec3, zeroVec1]
                          `shouldBe` [zeroVec2, zeroVec3, zeroVec1]
    it "passes non-zero through zero case" $ do
      backPropagate nonZeroVec1 zeroNet [zeroVec3, zeroVec1]
                          `shouldBe` [zeroVec2, zeroVec3, vector [1]]
    it "passes zero through simple case" $ do
      backPropagate zeroVec1 simpleNet [nonZeroVec2, vector [-2]]
                          `shouldBe` [zeroVec3, zeroVec2, zeroVec1]
    it "passes non-zero through simple case" $ do
      backPropagate nonZeroVec1 simpleNet [vector [-0.5,14], vector [4]]
                          `shouldBe` [vector [-0.5,1,1.5], vector [-2,0.5], nonZeroVec1]
