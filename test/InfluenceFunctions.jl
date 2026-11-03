for s in (:influence, :value_and_influence, :QuantileSpacing)
    @test Base.isexported(InfluenceFunctions, s)
end
@test_throws MethodError influence(maximum, [1.0, 2.0])             # an estimator without a method
@test_throws MethodError value_and_influence(maximum, [1.0, 2.0])
for s in (:fit_density, :HallSheatherBandwidth)                      # public, not exported
    @test !Base.isexported(InfluenceFunctions, s)
end
