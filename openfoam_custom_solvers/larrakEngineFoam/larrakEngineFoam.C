/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2026
-------------------------------------------------------------------------------
License
    This file is part of Larrak2's repo-owned OpenFOAM authority path.

Description
    Hybrid engine chemistry solver derived from the dynamic-mesh spray solver
    path with engine-style logging and a tracked repo-owned binary name.
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "dynamicFvMesh.H"
#include "turbulenceModel.H"
#include "basicSprayCloud.H"
#include "psiReactionThermo.H"
#include "CombustionModel.H"
#include "radiationModel.H"
#include "SLGThermo.H"
#include "pimpleControl.H"
#include "CorrectPhi.H"
#include "fvOptions.H"
#include "OFstream.H"
#include <algorithm>
#include <cmath>

namespace Foam
{

static IOdictionary readEngineGeometry(const Time& runTime)
{
    return IOdictionary
    (
        IOobject
        (
            "engineGeometry",
            runTime.constant(),
            runTime,
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
        )
    );
}


static scalar crankAngleDeg(const IOdictionary& engineGeometry, const Time& runTime)
{
    const scalar rpm =
        engineGeometry.lookupOrDefault<scalar>("rpm", 1500.0);
    const scalar initialCrankAngleDeg =
        engineGeometry.lookupOrDefault<scalar>("initialCrankAngleDeg", -180.0);

    return initialCrankAngleDeg + runTime.value()*rpm*360.0/60.0;
}


static void writeEngineSummary
(
    const Time& runTime,
    const scalar crankAngle,
    const volScalarField& p,
    const volScalarField& T,
    const volVectorField& U,
    const fvMesh& mesh
)
{
    const scalar meanPressure = p.weightedAverage(mesh.V()).value();
    const scalar meanTemperature = T.weightedAverage(mesh.V()).value();
    const tmp<volScalarField> velocityMagnitude = mag(U);
    const scalar meanVelocityMagnitude = velocityMagnitude().weightedAverage(mesh.V()).value();

    Info<< "Mean pressure: " << meanPressure << endl;
    Info<< "Mean temperature: " << meanTemperature << endl;
    Info<< "Mean |U|: " << meanVelocityMagnitude << endl;

    OFstream logSummaryFile
    (
        runTime.path()/("logSummary." + runTime.timeName() + ".dat")
    );

    logSummaryFile
        << "# crankAngleDeg meanPressurePa meanTemperatureK meanVelocityMagnitude\n"
        << crankAngle << tab
        << meanPressure << tab
        << meanTemperature << tab
        << meanVelocityMagnitude << nl;
}


static void stabilizeThermoState
(
    psiReactionThermo& thermo,
    const volScalarField& p,
    const IOdictionary& engineGeometry
)
{
    const scalar minTemperature =
        engineGeometry.lookupOrDefault<scalar>("minTemperatureK", 300.0);
    const scalar maxTemperature =
        engineGeometry.lookupOrDefault<scalar>("maxTemperatureK", 3500.0);
    const scalar maxThermoDeltaT =
        engineGeometry.lookupOrDefault<scalar>("maxThermoDeltaTK", GREAT);

    if (minTemperature >= maxTemperature)
    {
        FatalErrorInFunction
            << "Invalid thermo clamp range: minTemperatureK="
            << minTemperature << " maxTemperatureK=" << maxTemperature
            << exit(FatalError);
    }

    volScalarField& he = thermo.he();
    volScalarField& T = thermo.T();
    const dimensionedScalar Tmin("Tmin", dimTemperature, minTemperature);
    const dimensionedScalar Tmax("Tmax", dimTemperature, maxTemperature);

    tmp<volScalarField> tLowerT
    (
        volScalarField::New
        (
            "engineLowerTemperature",
            IOobject::NO_REGISTER,
            he.mesh(),
            Tmin
        )
    );
    tmp<volScalarField> tUpperT
    (
        volScalarField::New
        (
            "engineUpperTemperature",
            IOobject::NO_REGISTER,
            he.mesh(),
            Tmax
        )
    );

    volScalarField& lowerTField = tLowerT.ref();
    volScalarField& upperTField = tUpperT.ref();
    scalar limitedTemperatureWindows = 0;

    scalarField& lowerTCells = lowerTField.primitiveFieldRef();
    scalarField& upperTCells = upperTField.primitiveFieldRef();
    scalarField& TCells = T.primitiveFieldRef();
    forAll(TCells, celli)
    {
        const scalar boundedCurrentT = min(max(TCells[celli], minTemperature), maxTemperature);
        if (maxThermoDeltaT < GREAT/2)
        {
            lowerTCells[celli] = max(minTemperature, boundedCurrentT - maxThermoDeltaT);
            upperTCells[celli] = min(maxTemperature, boundedCurrentT + maxThermoDeltaT);
            if
            (
                mag(lowerTCells[celli] - minTemperature) > SMALL
             || mag(upperTCells[celli] - maxTemperature) > SMALL
            )
            {
                limitedTemperatureWindows += 1;
            }
        }
        else
        {
            lowerTCells[celli] = minTemperature;
            upperTCells[celli] = maxTemperature;
        }
        TCells[celli] = boundedCurrentT;
    }

    auto& lowerTBoundary = lowerTField.boundaryFieldRef();
    auto& upperTBoundary = upperTField.boundaryFieldRef();
    auto& TBoundary = T.boundaryFieldRef();
    forAll(TBoundary, patchi)
    {
        forAll(TBoundary[patchi], facei)
        {
            const scalar boundedCurrentT =
                min(max(TBoundary[patchi][facei], minTemperature), maxTemperature);
            if (maxThermoDeltaT < GREAT/2)
            {
                lowerTBoundary[patchi][facei] =
                    max(minTemperature, boundedCurrentT - maxThermoDeltaT);
                upperTBoundary[patchi][facei] =
                    min(maxTemperature, boundedCurrentT + maxThermoDeltaT);
                if
                (
                    mag(lowerTBoundary[patchi][facei] - minTemperature) > SMALL
                 || mag(upperTBoundary[patchi][facei] - maxTemperature) > SMALL
                )
                {
                    limitedTemperatureWindows += 1;
                }
            }
            else
            {
                lowerTBoundary[patchi][facei] = minTemperature;
                upperTBoundary[patchi][facei] = maxTemperature;
            }
            TBoundary[patchi][facei] = boundedCurrentT;
        }
    }

    const tmp<volScalarField> tMinHe = thermo.he(p, lowerTField);
    const tmp<volScalarField> tMaxHe = thermo.he(p, upperTField);
    const volScalarField& minHe = tMinHe();
    const volScalarField& maxHe = tMaxHe();

    scalar nClipped = 0;

    scalarField& heCells = he.primitiveFieldRef();
    const scalarField& minHeCells = minHe.primitiveField();
    const scalarField& maxHeCells = maxHe.primitiveField();
    forAll(heCells, celli)
    {
        const scalar original = heCells[celli];
        const scalar clipped = min(max(original, minHeCells[celli]), maxHeCells[celli]);
        if (clipped != heCells[celli])
        {
            heCells[celli] = clipped;
            TCells[celli] =
                clipped <= minHeCells[celli] ? lowerTCells[celli] : upperTCells[celli];
            nClipped += 1;
        }
        else
        {
            TCells[celli] = min(max(TCells[celli], lowerTCells[celli]), upperTCells[celli]);
        }
    }

    auto& heBoundary = he.boundaryFieldRef();
    const auto& minHeBoundary = minHe.boundaryField();
    const auto& maxHeBoundary = maxHe.boundaryField();
    forAll(heBoundary, patchi)
    {
        forAll(heBoundary[patchi], facei)
        {
            const scalar original = heBoundary[patchi][facei];
            const scalar clipped =
                min
                (
                    max(original, minHeBoundary[patchi][facei]),
                    maxHeBoundary[patchi][facei]
                );
            if (clipped != heBoundary[patchi][facei])
            {
                heBoundary[patchi][facei] = clipped;
                TBoundary[patchi][facei] =
                    clipped <= minHeBoundary[patchi][facei]
                  ? lowerTBoundary[patchi][facei]
                  : upperTBoundary[patchi][facei];
                nClipped += 1;
            }
            else
            {
                TBoundary[patchi][facei] =
                    min
                    (
                        max(TBoundary[patchi][facei], lowerTBoundary[patchi][facei]),
                        upperTBoundary[patchi][facei]
                    );
            }
        }
    }

    if (nClipped > 0)
    {
        Info<< "Clipped thermo energy state before correction: "
            << nClipped << " values into [Tmin, Tmax] = ["
            << minTemperature << ", " << maxTemperature << "]" << endl;
    }
    if (limitedTemperatureWindows > 0 && maxThermoDeltaT < GREAT/2)
    {
        Info<< "Limited thermo correction window with maxThermoDeltaTK="
            << maxThermoDeltaT << " on " << limitedTemperatureWindows
            << " values" << endl;
    }
}


static void stabilizePressureDensityState
(
    volScalarField& p,
    volScalarField& rho,
    const IOdictionary& engineGeometry
)
{
    const scalar minPressure =
        engineGeometry.lookupOrDefault<scalar>("minPressurePa", SMALL);
    const scalar minDensity =
        engineGeometry.lookupOrDefault<scalar>("minDensityKgM3", SMALL);

    scalar nPressureClipped = 0;
    scalar nDensityClipped = 0;

    scalarField& pCells = p.primitiveFieldRef();
    forAll(pCells, celli)
    {
        if (pCells[celli] < minPressure)
        {
            pCells[celli] = minPressure;
            nPressureClipped += 1;
        }
    }

    auto& pBoundary = p.boundaryFieldRef();
    forAll(pBoundary, patchi)
    {
        forAll(pBoundary[patchi], facei)
        {
            if (pBoundary[patchi][facei] < minPressure)
            {
                pBoundary[patchi][facei] = minPressure;
                nPressureClipped += 1;
            }
        }
    }

    scalarField& rhoCells = rho.primitiveFieldRef();
    forAll(rhoCells, celli)
    {
        if (rhoCells[celli] < minDensity)
        {
            rhoCells[celli] = minDensity;
            nDensityClipped += 1;
        }
    }

    auto& rhoBoundary = rho.boundaryFieldRef();
    forAll(rhoBoundary, patchi)
    {
        forAll(rhoBoundary[patchi], facei)
        {
            if (rhoBoundary[patchi][facei] < minDensity)
            {
                rhoBoundary[patchi][facei] = minDensity;
                nDensityClipped += 1;
            }
        }
    }

    if (nPressureClipped > 0)
    {
        Info<< "Clipped pressure floor: " << nPressureClipped
            << " values to minPressurePa=" << minPressure << endl;
    }
    if (nDensityClipped > 0)
    {
        Info<< "Clipped density floor: " << nDensityClipped
            << " values to minDensityKgM3=" << minDensity << endl;
    }
}


class RuntimeChemistryTable
{
    struct InterpolatedState
    {
        scalar qdot;
        scalar qdotTemperatureSensitivity;
        scalarList sourceTerms;
        scalarList diagJacobian;
    };

public:
    struct AuthorityMissSummary
    {
        label totalQueriedCells;
        label uncoveredCells;
        label trustRejectCells;
        scalar maxObservedUntrackedMassFraction;
        bool hasFirstTrustReject;
        word rejectVariable;
        word failureClass;
        word failureBranchId;
        scalar rejectExcess;
        scalar rejectNearestSampleDistance;
        scalarList rejectState;
        DynamicList<word> offendingVariables;
        DynamicList<label> offendingCounts;
        DynamicList<scalar> maxOutOfBoundByVariable;

        AuthorityMissSummary()
        :
            totalQueriedCells(0),
            uncoveredCells(0),
            trustRejectCells(0),
            maxObservedUntrackedMassFraction(0.0),
            hasFirstTrustReject(false),
            rejectVariable(word::null),
            failureClass(word::null),
            failureBranchId(word::null),
            rejectExcess(0.0),
            rejectNearestSampleDistance(0.0)
        {}

        void note(const word& variable, const scalar excess)
        {
            forAll(offendingVariables, index)
            {
                if (offendingVariables[index] == variable)
                {
                    offendingCounts[index] += 1;
                    maxOutOfBoundByVariable[index] =
                        max(maxOutOfBoundByVariable[index], excess);
                    return;
                }
            }
            offendingVariables.append(variable);
            offendingCounts.append(1);
            maxOutOfBoundByVariable.append(max(excess, scalar(0.0)));
        }

        void noteFirstTrustReject
        (
            const word& variable,
            const word& failureClassValue,
            const word& failureBranchIdValue,
            const scalar excess,
            const scalar nearestSampleDistance,
            const scalarList& state
        )
        {
            if (hasFirstTrustReject)
            {
                return;
            }
            hasFirstTrustReject = true;
            rejectVariable = variable;
            failureClass = failureClassValue;
            failureBranchId = failureBranchIdValue;
            rejectExcess = excess;
            rejectNearestSampleDistance = nearestSampleDistance;
            rejectState = state;
        }
    };

    struct CoverageCorpusRow
    {
        labelList coverageBucketKey;
        scalarList rawStateSum;
        scalarList transformedStateSum;
        wordList stageNames;
        label queryCount;
        label tableHitCount;
        label coverageRejectCount;
        label trustRejectCount;
        word worstRejectVariable;
        scalar worstRejectExcess;
        bool hasNearestSampleDistance;
        scalar nearestSampleDistanceMin;
        scalar nearestSampleDistanceMax;
        bool hasBestTrustRejectSnapshot;
        scalar bestTrustRejectSnapshotExcess;
        scalarList bestTrustRejectRawState;
        scalarList bestTrustRejectTransformedState;

        CoverageCorpusRow()
        :
            queryCount(0),
            tableHitCount(0),
            coverageRejectCount(0),
            trustRejectCount(0),
            worstRejectVariable(word::null),
            worstRejectExcess(0.0),
            hasNearestSampleDistance(false),
            nearestSampleDistanceMin(0.0),
            nearestSampleDistanceMax(0.0),
            hasBestTrustRejectSnapshot(false),
            bestTrustRejectSnapshotExcess(-1.0),
            bestTrustRejectRawState(),
            bestTrustRejectTransformedState()
        {}
    };

private:
    bool active_;
    word tableId_;
    word interpolationMethod_;
    word fallbackPolicy_;
    wordList stateVariables_;
    word balanceSpecies_;
    scalar maxUntrackedMassFraction_;
    label sampleCount_;
    scalarList stateScales_;
    wordList stateSpecies_;
    wordList speciesNames_;
    List<scalarList> axes_;
    List<scalarList> sampleStates_;
    boolList transformedStateFlags_;
    scalarList stateTransformFloors_;
    scalarList qdotValues_;
    scalarList qdotTemperatureSensitivity_;
    List<scalarList> sourceTermsBySpecies_;
    List<scalarList> diagJacobianBySpecies_;
    labelList stateSpeciesFieldIndices_;
    label balanceSpeciesFieldIndex_;
    labelList tableSpeciesFieldIndices_;
    label rbfNeighborCount_;
    scalar rbfEpsilon_;
    scalar rbfEnvelopeScale_;
    scalar rbfDiagEnvelopeScaleHO2_;
    scalar lookupCacheQuantization_;
    scalar coverageCorpusQuantization_;
    Switch coverageCorpusHighFidelityTrustSnapshots_;
    wordList coverageCorpusHighFidelityStageNames_;
    bool coverageCorpusHighFidelityUseAngleGate_;
    scalar coverageCorpusHighFidelityAngleMinDeg_;
    scalar coverageCorpusHighFidelityAngleMaxDeg_;
    scalar trustRegionMaxAbsSource_;
    scalar trustRegionMaxAbsJacobian_;
    scalar trustRegionMaxAbsQdot_;
    Switch skipStencilEnvelopeNonStateSpecies_;
    mutable label tableQueryCells_;
    mutable label tableHitCells_;
    mutable label fallbackTimestepCount_;
    mutable label interpolationCacheHits_;
    mutable label coverageRejectCells_;
    mutable label trustRegionRejectCells_;
    mutable HashTable<label, word> cacheIndexByKey_;
    mutable DynamicList<InterpolatedState> cacheEntries_;
    mutable HashTable<label, word> coverageRowIndexByKey_;
    mutable DynamicList<CoverageCorpusRow> coverageRows_;

    static label findFieldIndex(const PtrList<volScalarField>& Y, const word& speciesName)
    {
        forAll(Y, fieldi)
        {
            if (Y[fieldi].name() == speciesName)
            {
                return fieldi;
            }
        }
        return -1;
    }

    label nearestAxisIndex(const scalarList& axis, const scalar value) const
    {
        label nearestIndex = 0;
        scalar nearestDistance = mag(value - axis[0]);
        for (label i = 1; i < axis.size(); ++i)
        {
            const scalar distance = mag(value - axis[i]);
            if (distance < nearestDistance)
            {
                nearestDistance = distance;
                nearestIndex = i;
            }
        }
        return nearestIndex;
    }

    label flatIndex(const labelList& axisIndices) const
    {
        label index = 0;
        forAll(axisIndices, dimi)
        {
            index = index*axes_[dimi].size() + axisIndices[dimi];
        }
        return index;
    }

    static scalar sqrDistance(const scalarList& left, const scalarList& right)
    {
        scalar value = 0.0;
        forAll(left, dimi)
        {
            value += sqr(left[dimi] - right[dimi]);
        }
        return value;
    }

    scalar normalizedDistance(const scalarList& left, const scalarList& right) const
    {
        scalarList normalizedLeft(left.size(), 0.0);
        scalarList normalizedRight(right.size(), 0.0);
        forAll(left, dimi)
        {
            const scalar scale = max(stateScales_[dimi], SMALL);
            normalizedLeft[dimi] = left[dimi]/scale;
            normalizedRight[dimi] = right[dimi]/scale;
        }
        return sqrt(sqrDistance(normalizedLeft, normalizedRight));
    }

    static scalar gaussianKernel(const scalar distance, const scalar epsilon)
    {
        return exp(-sqr(max(epsilon, SMALL)*distance));
    }

    scalar transformStateValue(const label dimi, const scalar value) const
    {
        if (!transformedStateFlags_[dimi])
        {
            return value;
        }
        const scalar floor = max(stateTransformFloors_[dimi], scalar(1.0e-300));
        return log10(max(value, floor));
    }

    scalarList transformState(const scalarList& rawState) const
    {
        scalarList transformed(rawState);
        forAll(transformed, dimi)
        {
            transformed[dimi] = transformStateValue(dimi, transformed[dimi]);
        }
        return transformed;
    }

    static scalarList solveLinearSystem(List<scalarList> matrix, scalarList rhs)
    {
        const label size = matrix.size();
        for (label pivot = 0; pivot < size; ++pivot)
        {
            label bestRow = pivot;
            scalar bestValue = mag(matrix[pivot][pivot]);
            for (label row = pivot + 1; row < size; ++row)
            {
                const scalar candidate = mag(matrix[row][pivot]);
                if (candidate > bestValue)
                {
                    bestValue = candidate;
                    bestRow = row;
                }
            }
            if (bestRow != pivot)
            {
                Swap(matrix[pivot], matrix[bestRow]);
                Swap(rhs[pivot], rhs[bestRow]);
            }

            const scalar diagonal = matrix[pivot][pivot];
            if (mag(diagonal) < SMALL)
            {
                return rhs;
            }
            for (label row = pivot + 1; row < size; ++row)
            {
                const scalar factor = matrix[row][pivot]/diagonal;
                if (mag(factor) < SMALL)
                {
                    continue;
                }
                for (label col = pivot; col < size; ++col)
                {
                    matrix[row][col] -= factor*matrix[pivot][col];
                }
                rhs[row] -= factor*rhs[pivot];
            }
        }

        scalarList solution(size, 0.0);
        for (label row = size - 1; row >= 0; --row)
        {
            scalar residual = rhs[row];
            for (label col = row + 1; col < size; ++col)
            {
                residual -= matrix[row][col]*solution[col];
            }
            const scalar diagonal = matrix[row][row];
            solution[row] = mag(diagonal) < SMALL ? 0.0 : residual/diagonal;
        }
        return solution;
    }

    word cacheKey(const scalarList& queryState) const
    {
        word key;
        forAll(queryState, dimi)
        {
            const scalar quant = max(lookupCacheQuantization_, 1.0e-9)*max(stateScales_[dimi], scalar(1.0));
            const label bucket = label(std::llround(queryState[dimi]/quant));
            key += Foam::name(bucket);
            key += "_";
        }
        return key;
    }

    labelList coverageBucketKey(const scalarList& transformedState) const
    {
        labelList bucket(transformedState.size(), 0);
        forAll(transformedState, dimi)
        {
            const scalar quant =
                max(coverageCorpusQuantization_, 1.0e-9)
               *max(stateScales_[dimi], scalar(1.0));
            bucket[dimi] = label(std::llround(transformedState[dimi]/quant));
        }
        return bucket;
    }

    word coverageBucketWord(const scalarList& transformedState) const
    {
        const labelList bucket = coverageBucketKey(transformedState);
        word key;
        forAll(bucket, dimi)
        {
            key += Foam::name(bucket[dimi]);
            key += "_";
        }
        return key;
    }

    void noteCoverageObservation
    (
        const word& stageName,
        const scalarList& rawState,
        const scalarList& transformedState,
        const bool tableHit,
        const bool coverageReject,
        const bool trustReject,
        const word& rejectVariable,
        const scalar rejectExcess,
        const scalar nearestSampleDistance,
        const bool allowHighFidelityTrustSnapshot
    ) const
    {
        const word key = coverageBucketWord(transformedState);
        label rowIndex = -1;
        if (coverageRowIndexByKey_.found(key))
        {
            rowIndex = coverageRowIndexByKey_[key];
        }
        else
        {
            CoverageCorpusRow row;
            row.coverageBucketKey = coverageBucketKey(transformedState);
            row.rawStateSum.setSize(rawState.size(), 0.0);
            row.transformedStateSum.setSize(transformedState.size(), 0.0);
            rowIndex = coverageRows_.size();
            coverageRows_.append(row);
            coverageRowIndexByKey_.insert(key, rowIndex);
        }

        CoverageCorpusRow& row = coverageRows_[rowIndex];
        row.queryCount += 1;
        if (tableHit)
        {
            row.tableHitCount += 1;
        }
        if (coverageReject)
        {
            row.coverageRejectCount += 1;
        }
        if (trustReject)
        {
            row.trustRejectCount += 1;
        }
        forAll(rawState, dimi)
        {
            row.rawStateSum[dimi] += rawState[dimi];
            row.transformedStateSum[dimi] += transformedState[dimi];
        }
        if (stageName.size())
        {
            bool knownStage = false;
            forAll(row.stageNames, stagei)
            {
                if (row.stageNames[stagei] == stageName)
                {
                    knownStage = true;
                    break;
                }
            }
            if (!knownStage)
            {
                row.stageNames.append(stageName);
            }
        }
        if (rejectVariable.size() && rejectExcess >= row.worstRejectExcess)
        {
            row.worstRejectVariable = rejectVariable;
            row.worstRejectExcess = rejectExcess;
        }
        if (nearestSampleDistance >= 0.0 && std::isfinite(nearestSampleDistance))
        {
            if (!row.hasNearestSampleDistance)
            {
                row.hasNearestSampleDistance = true;
                row.nearestSampleDistanceMin = nearestSampleDistance;
                row.nearestSampleDistanceMax = nearestSampleDistance;
            }
            else
            {
                row.nearestSampleDistanceMin =
                    min(row.nearestSampleDistanceMin, nearestSampleDistance);
                row.nearestSampleDistanceMax =
                    max(row.nearestSampleDistanceMax, nearestSampleDistance);
            }
        }
        if
        (
            allowHighFidelityTrustSnapshot
         && trustReject
         && rejectExcess > row.bestTrustRejectSnapshotExcess
        )
        {
            row.hasBestTrustRejectSnapshot = true;
            row.bestTrustRejectSnapshotExcess = rejectExcess;
            row.bestTrustRejectRawState = rawState;
            row.bestTrustRejectTransformedState = transformedState;
        }
    }

    static bool isLookupMode(const word& mode)
    {
        return mode == "lookupTable"
            || mode == "lookupTablePermissive"
            || mode == "lookupTableStrict";
    }

    static bool isStrictLookupMode(const word& mode)
    {
        return mode == "lookupTableStrict";
    }

    bool interpolateState
    (
        const scalarList& queryState,
        InterpolatedState& interpolated,
        word& rejectVariable,
        scalar& rejectExcess,
        word& failureClass,
        word& failureBranchId,
        scalar& rejectNearestSampleDistance
    ) const
    {
        rejectVariable = word("");
        rejectExcess = 0.0;
        failureClass = word("");
        failureBranchId = word("");
        rejectNearestSampleDistance = 0.0;
        const word key = cacheKey(queryState);
        if (cacheIndexByKey_.found(key))
        {
            interpolationCacheHits_ += 1;
            interpolated = cacheEntries_[cacheIndexByKey_[key]];
            return true;
        }

        label exactIndex = -1;
        scalar exactDistance = GREAT;
        List<Tuple2<scalar, label>> distanceIndex(sampleCount_);
        forAll(sampleStates_, samplei)
        {
            const scalar distance = normalizedDistance(queryState, sampleStates_[samplei]);
            distanceIndex[samplei] = Tuple2<scalar, label>(distance, samplei);
            if (distance < exactDistance)
            {
                exactDistance = distance;
                exactIndex = samplei;
            }
        }

        if (exactIndex >= 0 && exactDistance <= 1.0e-12)
        {
            interpolated.qdot = qdotValues_[exactIndex];
            interpolated.qdotTemperatureSensitivity = qdotTemperatureSensitivity_[exactIndex];
            interpolated.sourceTerms.setSize(speciesNames_.size(), 0.0);
            interpolated.diagJacobian.setSize(speciesNames_.size(), 0.0);
            forAll(speciesNames_, speciesi)
            {
                interpolated.sourceTerms[speciesi] = sourceTermsBySpecies_[speciesi][exactIndex];
                interpolated.diagJacobian[speciesi] = diagJacobianBySpecies_[speciesi][exactIndex];
            }
            cacheIndexByKey_.insert(key, cacheEntries_.size());
            cacheEntries_.append(interpolated);
            return true;
        }

        std::sort
        (
            distanceIndex.begin(),
            distanceIndex.end(),
            [](const Tuple2<scalar, label>& left, const Tuple2<scalar, label>& right)
            {
                return left.first() < right.first();
            }
        );
        if (sampleCount_ > 0)
        {
            rejectNearestSampleDistance = distanceIndex[0].first();
        }

        const label stencil = max(label(2), min(rbfNeighborCount_, sampleCount_));
        List<scalarList> kernel(stencil);
        scalarList rhs(stencil, 0.0);
        scalarList stencilQdot(stencil, 0.0);
        scalarList stencilQdotDT(stencil, 0.0);
        for (label row = 0; row < stencil; ++row)
        {
            kernel[row].setSize(stencil, 0.0);
            const label rowIndex = distanceIndex[row].second();
            rhs[row] = gaussianKernel(distanceIndex[row].first(), rbfEpsilon_);
            stencilQdot[row] = qdotValues_[rowIndex];
            stencilQdotDT[row] = qdotTemperatureSensitivity_[rowIndex];
            for (label col = 0; col < stencil; ++col)
            {
                const label colIndex = distanceIndex[col].second();
                kernel[row][col] = gaussianKernel
                (
                    normalizedDistance(sampleStates_[rowIndex], sampleStates_[colIndex]),
                    rbfEpsilon_
                );
                if (row == col)
                {
                    kernel[row][col] += 1.0e-10;
                }
            }
        }
        const scalarList weights = solveLinearSystem(kernel, rhs);
        const scalar envelopeScale = max(rbfEnvelopeScale_, scalar(0.0));

        interpolated.qdot = 0.0;
        interpolated.qdotTemperatureSensitivity = 0.0;
        interpolated.sourceTerms.setSize(speciesNames_.size(), 0.0);
        interpolated.diagJacobian.setSize(speciesNames_.size(), 0.0);
        for (label weighti = 0; weighti < stencil; ++weighti)
        {
            const scalar weight = weights[weighti];
            const label samplei = distanceIndex[weighti].second();
            interpolated.qdot += weight*qdotValues_[samplei];
            interpolated.qdotTemperatureSensitivity += weight*qdotTemperatureSensitivity_[samplei];
            forAll(speciesNames_, speciesi)
            {
                interpolated.sourceTerms[speciesi] += weight*sourceTermsBySpecies_[speciesi][samplei];
                interpolated.diagJacobian[speciesi] += weight*diagJacobianBySpecies_[speciesi][samplei];
            }
        }

        scalar qdotMin = GREAT;
        scalar qdotMax = -GREAT;
        scalar qdotDTMin = GREAT;
        scalar qdotDTMax = -GREAT;
        forAll(stencilQdot, index)
        {
            qdotMin = min(qdotMin, stencilQdot[index]);
            qdotMax = max(qdotMax, stencilQdot[index]);
            qdotDTMin = min(qdotDTMin, stencilQdotDT[index]);
            qdotDTMax = max(qdotDTMax, stencilQdotDT[index]);
        }
        const scalar qdotAmp = max(max(mag(qdotMin), mag(qdotMax)), scalar(1.0e-12));
        const scalar qdotSpan = mag(qdotMax - qdotMin);
        const scalar qdotMargin = envelopeScale*(qdotAmp + qdotSpan);
        const scalar qdotDTAmp = max(max(mag(qdotDTMin), mag(qdotDTMax)), scalar(1.0e-12));
        const scalar qdotDTSpan = mag(qdotDTMax - qdotDTMin);
        const scalar qdotDTMargin = envelopeScale*(qdotDTAmp + qdotDTSpan);
        if (!skipStencilEnvelopeNonStateSpecies_)
        {
            if (interpolated.qdot < qdotMin - qdotMargin || interpolated.qdot > qdotMax + qdotMargin)
            {
                trustRegionRejectCells_ += 1;
                rejectVariable = "Qdot";
                failureClass = "qdot";
                failureBranchId = "default";
                rejectExcess =
                    max(qdotMin - interpolated.qdot - qdotMargin, interpolated.qdot - qdotMax - qdotMargin);
                return false;
            }
            if
            (
                interpolated.qdotTemperatureSensitivity < qdotDTMin - qdotDTMargin
             || interpolated.qdotTemperatureSensitivity > qdotDTMax + qdotDTMargin
            )
            {
                trustRegionRejectCells_ += 1;
                rejectVariable = "QdotTemperatureSensitivity";
                failureClass = "qdot";
                failureBranchId = "default";
                rejectExcess =
                    max
                    (
                        qdotDTMin - interpolated.qdotTemperatureSensitivity - qdotDTMargin,
                        interpolated.qdotTemperatureSensitivity - qdotDTMax - qdotDTMargin
                    );
                return false;
            }
        }

        if
        (
            !std::isfinite(interpolated.qdot)
         || mag(interpolated.qdot) > trustRegionMaxAbsQdot_
         || !std::isfinite(interpolated.qdotTemperatureSensitivity)
         || mag(interpolated.qdotTemperatureSensitivity) > trustRegionMaxAbsQdot_
        )
        {
            trustRegionRejectCells_ += 1;
            rejectVariable = "Qdot";
            failureClass = "qdot";
            failureBranchId = "default";
            rejectExcess = max(mag(interpolated.qdot) - trustRegionMaxAbsQdot_, scalar(0.0));
            return false;
        }
        forAll(interpolated.sourceTerms, speciesi)
        {
            scalar stencilSourceMin = GREAT;
            scalar stencilSourceMax = -GREAT;
            scalar stencilDiagMin = GREAT;
            scalar stencilDiagMax = -GREAT;
            for (label weighti = 0; weighti < stencil; ++weighti)
            {
                const label samplei = distanceIndex[weighti].second();
                stencilSourceMin = min(stencilSourceMin, sourceTermsBySpecies_[speciesi][samplei]);
                stencilSourceMax = max(stencilSourceMax, sourceTermsBySpecies_[speciesi][samplei]);
                stencilDiagMin = min(stencilDiagMin, diagJacobianBySpecies_[speciesi][samplei]);
                stencilDiagMax = max(stencilDiagMax, diagJacobianBySpecies_[speciesi][samplei]);
            }
            const scalar sourceAmp =
                max(max(mag(stencilSourceMin), mag(stencilSourceMax)), scalar(1.0e-12));
            const scalar sourceSpan = mag(stencilSourceMax - stencilSourceMin);
            const scalar sourceStencilScale = sourceAmp + sourceSpan;
            scalar sourceMargin = envelopeScale*sourceStencilScale;
            const scalar diagAmp =
                max(max(mag(stencilDiagMin), mag(stencilDiagMax)), scalar(1.0e-12));
            const scalar diagSpan = mag(stencilDiagMax - stencilDiagMin);
            scalar diagMargin = envelopeScale*(diagAmp + diagSpan);
            bool speciesOnStateAxes = false;
            forAll(stateSpecies_, stateSpeciesi)
            {
                if (stateSpecies_[stateSpeciesi] == speciesNames_[speciesi])
                {
                    speciesOnStateAxes = true;
                    break;
                }
            }
            const bool enforceStencilEnvelope =
                !skipStencilEnvelopeNonStateSpecies_ || speciesOnStateAxes;
            if (
                enforceStencilEnvelope
             && speciesOnStateAxes
             && speciesNames_[speciesi] == word("H")
             && sourceStencilScale < scalar(1.0e-06)
            )
            {
                sourceMargin = max(sourceMargin, envelopeScale*scalar(8.0e-08));
            }
            else if (enforceStencilEnvelope && speciesOnStateAxes && sourceStencilScale < scalar(1.0e-07))
            {
                sourceMargin = max(sourceMargin, envelopeScale*scalar(3.5e-09));
            }
            if
            (
                speciesOnStateAxes
             && speciesNames_[speciesi] == word("HO2")
             && rbfDiagEnvelopeScaleHO2_ > scalar(1.0e-12)
            )
            {
                diagMargin *= rbfDiagEnvelopeScaleHO2_;
            }
            if
            (
                enforceStencilEnvelope
             && (
                    interpolated.sourceTerms[speciesi] < stencilSourceMin - sourceMargin
                 || interpolated.sourceTerms[speciesi] > stencilSourceMax + sourceMargin
                )
            )
            {
                trustRegionRejectCells_ += 1;
                rejectVariable = speciesNames_[speciesi];
                failureClass = "same_sign_overshoot";
                failureBranchId = "default";
                rejectExcess =
                    max
                    (
                        stencilSourceMin - interpolated.sourceTerms[speciesi] - sourceMargin,
                        interpolated.sourceTerms[speciesi] - stencilSourceMax - sourceMargin
                    );
                return false;
            }
            if
            (
                enforceStencilEnvelope
             && (
                    interpolated.diagJacobian[speciesi] < stencilDiagMin - diagMargin
                 || interpolated.diagJacobian[speciesi] > stencilDiagMax + diagMargin
                )
            )
            {
                trustRegionRejectCells_ += 1;
                rejectVariable = speciesNames_[speciesi] + "_diag";
                failureClass = "same_sign_overshoot";
                failureBranchId = "default";
                rejectExcess =
                    max
                    (
                        stencilDiagMin - interpolated.diagJacobian[speciesi] - diagMargin,
                        interpolated.diagJacobian[speciesi] - stencilDiagMax - diagMargin
                    );
                return false;
            }
            if
            (
                !std::isfinite(interpolated.sourceTerms[speciesi])
             || !std::isfinite(interpolated.diagJacobian[speciesi])
            )
            {
                trustRegionRejectCells_ += 1;
                rejectVariable = speciesNames_[speciesi];
                failureClass = "same_sign_overshoot";
                failureBranchId = "default";
                rejectExcess = GREAT;
                return false;
            }
            const bool enforceAbsTrustCap =
                !skipStencilEnvelopeNonStateSpecies_ || speciesOnStateAxes;
            if
            (
                enforceAbsTrustCap
             && (
                    mag(interpolated.sourceTerms[speciesi]) > trustRegionMaxAbsSource_
                 || mag(interpolated.diagJacobian[speciesi]) > trustRegionMaxAbsJacobian_
                )
            )
            {
                trustRegionRejectCells_ += 1;
                rejectVariable = speciesNames_[speciesi];
                failureClass = "same_sign_overshoot";
                failureBranchId = "default";
                rejectExcess =
                    max
                    (
                        max(mag(interpolated.sourceTerms[speciesi]) - trustRegionMaxAbsSource_, scalar(0.0)),
                        max(mag(interpolated.diagJacobian[speciesi]) - trustRegionMaxAbsJacobian_, scalar(0.0))
                    );
                return false;
            }
        }

        cacheIndexByKey_.insert(key, cacheEntries_.size());
        cacheEntries_.append(interpolated);
        return true;
    }

public:
    RuntimeChemistryTable
    (
        const fvMesh& mesh,
        const PtrList<volScalarField>& Y
    )
    :
        active_(false),
        tableId_(""),
        interpolationMethod_("local_rbf"),
        fallbackPolicy_("fullReducedKinetics"),
        stateVariables_(),
        balanceSpecies_("N2"),
        maxUntrackedMassFraction_(0.02),
        sampleCount_(0),
        transformedStateFlags_(),
        stateTransformFloors_(),
        balanceSpeciesFieldIndex_(-1),
        rbfNeighborCount_(8),
        rbfEpsilon_(1.0),
        rbfEnvelopeScale_(0.1),
        rbfDiagEnvelopeScaleHO2_(1.0),
        lookupCacheQuantization_(0.0025),
        coverageCorpusQuantization_(0.0025),
        coverageCorpusHighFidelityTrustSnapshots_(false),
        coverageCorpusHighFidelityStageNames_(),
        coverageCorpusHighFidelityUseAngleGate_(false),
        coverageCorpusHighFidelityAngleMinDeg_(-GREAT),
        coverageCorpusHighFidelityAngleMaxDeg_(GREAT),
        trustRegionMaxAbsSource_(1.0e12),
        trustRegionMaxAbsJacobian_(1.0e12),
        trustRegionMaxAbsQdot_(1.0e15),
        skipStencilEnvelopeNonStateSpecies_(false),
        tableQueryCells_(0),
        tableHitCells_(0),
        fallbackTimestepCount_(0),
        interpolationCacheHits_(0),
        coverageRejectCells_(0),
        trustRegionRejectCells_(0)
    {
        IOobject tableObject
        (
            "runtimeChemistryTable",
            mesh.time().constant(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE
        );

        if (!tableObject.typeHeaderOk<IOdictionary>(true))
        {
            return;
        }

        IOdictionary tableDict(tableObject);
        active_ = tableDict.lookupOrDefault<Switch>("active", false);
        if (!active_)
        {
            return;
        }

        tableId_ = tableDict.lookupOrDefault<word>("tableId", "runtimeChemistryTable");
        interpolationMethod_ =
            tableDict.lookupOrDefault<word>("interpolation", "local_rbf");
        fallbackPolicy_ =
            tableDict.lookupOrDefault<word>("fallbackPolicy", "fullReducedKinetics");
        maxUntrackedMassFraction_ =
            tableDict.lookupOrDefault<scalar>("maxUntrackedMassFraction", 0.02);
        balanceSpecies_ =
            tableDict.lookupOrDefault<word>("balanceSpecies", "N2");
        sampleCount_ =
            tableDict.lookupOrDefault<label>("sampleCount", 0);
        stateVariables_ = wordList(tableDict.lookup("stateVariables"));
        stateScales_ = scalarList(tableDict.lookup("stateScales"));
        rbfNeighborCount_ =
            tableDict.lookupOrDefault<label>("rbfNeighborCount", 8);
        rbfEpsilon_ =
            tableDict.lookupOrDefault<scalar>("rbfEpsilon", 1.0);
        rbfEnvelopeScale_ =
            tableDict.lookupOrDefault<scalar>("rbfEnvelopeScale", 0.1);
        rbfDiagEnvelopeScaleHO2_ =
            tableDict.lookupOrDefault<scalar>("rbfDiagEnvelopeScaleHO2", 1.0);
        lookupCacheQuantization_ =
            tableDict.lookupOrDefault<scalar>("lookupCacheQuantization", 0.0025);
        coverageCorpusQuantization_ =
            tableDict.found("coverageCorpusQuantization")
          ? readScalar(tableDict.lookup("coverageCorpusQuantization"))
          : lookupCacheQuantization_;
        coverageCorpusHighFidelityTrustSnapshots_ =
            tableDict.lookupOrDefault<Switch>(
                "coverageCorpusHighFidelityTrustSnapshots",
                false
            );
        coverageCorpusHighFidelityStageNames_ =
            tableDict.lookupOrDefault<wordList>(
                "coverageCorpusHighFidelityStageNames",
                wordList(0)
            );
        coverageCorpusHighFidelityUseAngleGate_ = false;
        coverageCorpusHighFidelityAngleMinDeg_ = -GREAT;
        coverageCorpusHighFidelityAngleMaxDeg_ = GREAT;
        if
        (
            tableDict.found("coverageCorpusHighFidelityAngleMinDeg")
         && tableDict.found("coverageCorpusHighFidelityAngleMaxDeg")
        )
        {
            coverageCorpusHighFidelityUseAngleGate_ = true;
            coverageCorpusHighFidelityAngleMinDeg_ = readScalar(
                tableDict.lookup("coverageCorpusHighFidelityAngleMinDeg")
            );
            coverageCorpusHighFidelityAngleMaxDeg_ = readScalar(
                tableDict.lookup("coverageCorpusHighFidelityAngleMaxDeg")
            );
        }
        trustRegionMaxAbsSource_ =
            tableDict.lookupOrDefault<scalar>("trustRegionMaxAbsSource", 1.0e12);
        trustRegionMaxAbsJacobian_ =
            tableDict.lookupOrDefault<scalar>("trustRegionMaxAbsJacobian", 1.0e12);
        trustRegionMaxAbsQdot_ =
            tableDict.lookupOrDefault<scalar>("trustRegionMaxAbsQdot", 1.0e15);
        skipStencilEnvelopeNonStateSpecies_ =
            tableDict.lookupOrDefault<Switch>("skipStencilEnvelopeNonStateSpecies", false);
        stateSpecies_ = wordList(tableDict.lookup("stateSpecies"));
        speciesNames_ = wordList(tableDict.lookup("speciesNames"));
        sampleStates_ = List<scalarList>(tableDict.lookup("sampleStates"));
        transformedStateFlags_.setSize(stateVariables_.size(), false);
        stateTransformFloors_.setSize(stateVariables_.size(), 0.0);
        wordList transformedStateVariables
        (
            tableDict.lookupOrDefault<wordList>("transformedStateVariables", wordList(0))
        );
        const bool hasStateTransformFloors = tableDict.found("stateTransformFloors");

        const dictionary& axesDict = tableDict.subDict("axes");
        axes_.setSize(stateSpecies_.size() + 2);
        axes_[0] = scalarList(axesDict.lookup("Temperature"));
        axes_[1] = scalarList(axesDict.lookup("Pressure"));
        forAll(stateSpecies_, speciesi)
        {
            axes_[speciesi + 2] = scalarList(axesDict.lookup(stateSpecies_[speciesi]));
        }
        if (stateVariables_.size() != axes_.size())
        {
            FatalErrorInFunction
                << "Runtime chemistry state variable count mismatch: "
                << stateVariables_.size() << " vs " << axes_.size()
                << exit(FatalError);
        }
        forAll(stateVariables_, dimi)
        {
            const word& variable = stateVariables_[dimi];
            transformedStateFlags_[dimi] = false;
            forAll(transformedStateVariables, trani)
            {
                if (transformedStateVariables[trani] == variable)
                {
                    transformedStateFlags_[dimi] = true;
                    break;
                }
            }
            stateTransformFloors_[dimi] =
                hasStateTransformFloors
              ? tableDict.subDict("stateTransformFloors").lookupOrDefault<scalar>(variable, 0.0)
              : 0.0;
        }

        qdotValues_ = scalarList(tableDict.lookup("qdot"));
        qdotTemperatureSensitivity_ = scalarList(tableDict.lookup("qdotTemperatureSensitivity"));
        const dictionary& sourceTermsDict = tableDict.subDict("sourceTerms");
        const dictionary& diagJacobianDict = tableDict.subDict("diagSourceJacobian");
        sourceTermsBySpecies_.setSize(speciesNames_.size());
        diagJacobianBySpecies_.setSize(speciesNames_.size());
        forAll(speciesNames_, speciesi)
        {
            sourceTermsBySpecies_[speciesi] =
                scalarList(sourceTermsDict.lookup(speciesNames_[speciesi]));
            diagJacobianBySpecies_[speciesi] =
                scalarList(diagJacobianDict.lookup(speciesNames_[speciesi]));
        }

        const label expectedPointCount = qdotValues_.size();
        if (sampleCount_ <= 0)
        {
            sampleCount_ = expectedPointCount;
        }
        forAll(axes_, dimi)
        {
            if (axes_[dimi].size() < 2)
            {
                FatalErrorInFunction
                    << "Runtime chemistry table axis " << dimi
                    << " must define at least two points" << exit(FatalError);
            }
        }
        forAll(sourceTermsBySpecies_, speciesi)
        {
            if (sourceTermsBySpecies_[speciesi].size() != expectedPointCount)
            {
                FatalErrorInFunction
                    << "Runtime chemistry source term size mismatch for species "
                    << speciesNames_[speciesi] << exit(FatalError);
            }
            if (diagJacobianBySpecies_[speciesi].size() != expectedPointCount)
            {
                FatalErrorInFunction
                    << "Runtime chemistry Jacobian size mismatch for species "
                    << speciesNames_[speciesi] << exit(FatalError);
            }
        }
        if (sampleStates_.size() != expectedPointCount)
        {
            FatalErrorInFunction
                << "Runtime chemistry sample state count mismatch: "
                << sampleStates_.size() << " vs " << expectedPointCount
                << exit(FatalError);
        }
        if (qdotTemperatureSensitivity_.size() != expectedPointCount)
        {
            FatalErrorInFunction
                << "Runtime chemistry qdot temperature sensitivity size mismatch"
                << exit(FatalError);
        }

        stateSpeciesFieldIndices_.setSize(stateSpecies_.size(), -1);
        forAll(stateSpecies_, speciesi)
        {
            stateSpeciesFieldIndices_[speciesi] =
                findFieldIndex(Y, stateSpecies_[speciesi]);
        }
        balanceSpeciesFieldIndex_ = findFieldIndex(Y, balanceSpecies_);
        tableSpeciesFieldIndices_.setSize(speciesNames_.size(), -1);
        forAll(speciesNames_, speciesi)
        {
            tableSpeciesFieldIndices_[speciesi] =
                findFieldIndex(Y, speciesNames_[speciesi]);
        }
    }

    bool available() const
    {
        return active_;
    }

    bool enabled(const IOdictionary& engineGeometry) const
    {
        const word mode =
            engineGeometry.lookupOrDefault<word>("runtimeChemistryMode", "fullReducedKinetics");
        return active_ && isLookupMode(mode);
    }

    bool strictMode(const IOdictionary& engineGeometry) const
    {
        const word mode =
            engineGeometry.lookupOrDefault<word>("runtimeChemistryMode", "fullReducedKinetics");
        const Switch strictFlag =
            engineGeometry.lookupOrDefault<Switch>("runtimeChemistryStrict", isStrictLookupMode(mode));
        return active_ && (isStrictLookupMode(mode) || strictFlag);
    }

    bool abortOnAuthorityMiss(const IOdictionary& engineGeometry) const
    {
        return engineGeometry.lookupOrDefault<Switch>("runtimeChemistryAbortOnAuthorityMiss", strictMode(engineGeometry));
    }

    const word& tableId() const
    {
        return tableId_;
    }

    const word& fallbackPolicy() const
    {
        return fallbackPolicy_;
    }

    bool populate
    (
        const Time& runTime,
        const IOdictionary& engineGeometry,
        const PtrList<volScalarField>& Y,
        const volScalarField& p,
        const volScalarField& T,
        PtrList<volScalarField>& runtimeSources,
        PtrList<volScalarField>& runtimeDiagSources,
        volScalarField& runtimeQdot,
        volScalarField& runtimeQdotTemperatureSensitivity,
        AuthorityMissSummary& missSummary
    ) const
    {
        missSummary = AuthorityMissSummary();
        if (!enabled(engineGeometry))
        {
            return false;
        }

        const word runtimeStageName =
            engineGeometry.lookupOrDefault<word>("runtimeChemistryStageName", "");
        const scalar crankAngleDegValue = crankAngleDeg(engineGeometry, runTime);
        bool angleOk =
            !coverageCorpusHighFidelityUseAngleGate_
         || (
                crankAngleDegValue >= coverageCorpusHighFidelityAngleMinDeg_
             && crankAngleDegValue <= coverageCorpusHighFidelityAngleMaxDeg_
            );
        bool stageOk = coverageCorpusHighFidelityStageNames_.empty();
        if (!stageOk)
        {
            forAll(coverageCorpusHighFidelityStageNames_, stagei)
            {
                if (coverageCorpusHighFidelityStageNames_[stagei] == runtimeStageName)
                {
                    stageOk = true;
                    break;
                }
            }
        }
        const bool allowHighFidelityTrustSnapshot =
            coverageCorpusHighFidelityTrustSnapshots_ && angleOk && stageOk;

        const scalar permittedUntracked =
            engineGeometry.lookupOrDefault<scalar>
            (
                "runtimeChemistryMaxUntrackedMassFraction",
                maxUntrackedMassFraction_
            );

        scalarField& runtimeQdotCells = runtimeQdot.primitiveFieldRef();
        scalarField& runtimeQdotTemperatureSensitivityCells =
            runtimeQdotTemperatureSensitivity.primitiveFieldRef();
        const scalarField& TCells = T.primitiveField();
        const scalarField& pCells = p.primitiveField();
        tableQueryCells_ += TCells.size();
        missSummary.totalQueriedCells = TCells.size();

        // Note: coverage reject / trust reject paths return false after the first failing cell,
        // so only one corpus observation is recorded per timestep along those branches.

        forAll(TCells, celli)
        {
            scalarList rawQueryState(axes_.size(), 0.0);
            const scalar stateT = TCells[celli];
            const scalar stateP = pCells[celli];
            rawQueryState[0] = stateT;
            rawQueryState[1] = stateP;

            bool coverageReject = false;
            word coverageRejectVariable(word::null);
            scalar coverageRejectExcess = 0.0;

            if
            (
                stateT < axes_[0][0] || stateT > axes_[0][axes_[0].size() - 1]
            )
            {
                coverageReject = true;
                coverageRejectVariable = "Temperature";
                coverageRejectExcess =
                    stateT < axes_[0][0]
                  ? axes_[0][0] - stateT
                  : stateT - axes_[0][axes_[0].size() - 1];
            }
            if
            (
                !coverageReject
             && (stateP < axes_[1][0] || stateP > axes_[1][axes_[1].size() - 1])
            )
            {
                coverageReject = true;
                coverageRejectVariable = "Pressure";
                coverageRejectExcess =
                    stateP < axes_[1][0]
                  ? axes_[1][0] - stateP
                  : stateP - axes_[1][axes_[1].size() - 1];
            }

            scalar trackedMassFraction = 0.0;
            forAll(stateSpeciesFieldIndices_, speciesi)
            {
                const label fieldIndex = stateSpeciesFieldIndices_[speciesi];
                const scalar speciesValue =
                    fieldIndex >= 0
                  ? Y[fieldIndex].primitiveField()[celli]
                  : 0.0;
                trackedMassFraction += speciesValue;
                rawQueryState[speciesi + 2] = speciesValue;
                if
                (
                    !coverageReject
                 && (
                        speciesValue < axes_[speciesi + 2][0]
                     || speciesValue > axes_[speciesi + 2][axes_[speciesi + 2].size() - 1]
                    )
                )
                {
                    coverageReject = true;
                    coverageRejectVariable = stateSpecies_[speciesi];
                    coverageRejectExcess =
                        speciesValue < axes_[speciesi + 2][0]
                      ? axes_[speciesi + 2][0] - speciesValue
                      : speciesValue - axes_[speciesi + 2][axes_[speciesi + 2].size() - 1];
                }
            }

            scalar balanceMassFraction = 0.0;
            if (balanceSpeciesFieldIndex_ >= 0)
            {
                balanceMassFraction = Y[balanceSpeciesFieldIndex_].primitiveField()[celli];
            }

            scalar untrackedMassFraction = 1.0 - trackedMassFraction - balanceMassFraction;
            if (untrackedMassFraction < 0.0)
            {
                untrackedMassFraction = 0.0;
            }
            missSummary.maxObservedUntrackedMassFraction =
                max(missSummary.maxObservedUntrackedMassFraction, untrackedMassFraction);
            if (!coverageReject && untrackedMassFraction > permittedUntracked)
            {
                coverageReject = true;
                coverageRejectVariable = "untrackedMassFraction";
                coverageRejectExcess = untrackedMassFraction - permittedUntracked;
            }

            const scalarList queryState = transformState(rawQueryState);
            if (coverageReject)
            {
                missSummary.uncoveredCells += 1;
                coverageRejectCells_ += 1;
                missSummary.note(coverageRejectVariable, coverageRejectExcess);
                noteCoverageObservation(
                    runtimeStageName,
                    rawQueryState,
                    queryState,
                    false,
                    true,
                    false,
                    coverageRejectVariable,
                    coverageRejectExcess,
                    -1.0,
                    false
                );
                return false;
            }
            InterpolatedState interpolated;
            word rejectVariable;
            scalar rejectExcess = 0.0;
            word failureClass;
            word failureBranchId;
            scalar rejectNearestSampleDistance = 0.0;
            if
            (
                !interpolateState
                (
                    queryState,
                    interpolated,
                    rejectVariable,
                    rejectExcess,
                    failureClass,
                    failureBranchId,
                    rejectNearestSampleDistance
                )
            )
            {
                missSummary.trustRejectCells += 1;
                missSummary.note(
                    rejectVariable.size() == 0 ? word("interpolation") : rejectVariable,
                    rejectExcess
                );
                missSummary.noteFirstTrustReject
                (
                    rejectVariable.size() == 0 ? word("interpolation") : rejectVariable,
                    failureClass.size() == 0 ? word("undetailed_authority_miss") : failureClass,
                    failureBranchId.size() == 0 ? word("default") : failureBranchId,
                    rejectExcess,
                    rejectNearestSampleDistance,
                    rawQueryState
                );
                noteCoverageObservation(
                    runtimeStageName,
                    rawQueryState,
                    queryState,
                    false,
                    false,
                    true,
                    rejectVariable.size() == 0 ? word("interpolation") : rejectVariable,
                    rejectExcess,
                    rejectNearestSampleDistance,
                    allowHighFidelityTrustSnapshot
                );
                return false;
            }
            noteCoverageObservation(
                runtimeStageName,
                rawQueryState,
                queryState,
                true,
                false,
                false,
                word::null,
                0.0,
                rejectNearestSampleDistance,
                false
            );
            tableHitCells_ += 1;
            runtimeQdotCells[celli] = interpolated.qdot;
            runtimeQdotTemperatureSensitivityCells[celli] =
                interpolated.qdotTemperatureSensitivity;
            forAll(tableSpeciesFieldIndices_, speciesi)
            {
                const label fieldIndex = tableSpeciesFieldIndices_[speciesi];
                if (fieldIndex >= 0)
                {
                    runtimeSources[fieldIndex].primitiveFieldRef()[celli] =
                        interpolated.sourceTerms[speciesi];
                    runtimeDiagSources[fieldIndex].primitiveFieldRef()[celli] =
                        interpolated.diagJacobian[speciesi];
                }
            }
        }

        return true;
    }

    void writeAuthorityMiss
    (
        const Time& runTime,
        const IOdictionary& engineGeometry,
        const AuthorityMissSummary& missSummary
    ) const
    {
        if (!active_)
        {
            return;
        }
        OFstream missFile(runTime.path()/"runtimeChemistryAuthorityMiss.json");
        missFile
            << "{\n"
            << "  \"stage_name\": \"" << engineGeometry.lookupOrDefault<word>("runtimeChemistryStageName", "") << "\",\n"
            << "  \"checkpoint_time_s\": " << runTime.value() << ",\n"
            << "  \"crank_angle_deg\": " << crankAngleDeg(engineGeometry, runTime) << ",\n"
            << "  \"total_queried_cells\": " << missSummary.totalQueriedCells << ",\n"
            << "  \"uncovered_cell_count\": " << missSummary.uncoveredCells << ",\n"
            << "  \"trust_reject_cell_count\": " << missSummary.trustRejectCells << ",\n"
            << "  \"max_untracked_mass_fraction\": " << missSummary.maxObservedUntrackedMassFraction << ",\n"
            << "  \"table_id\": \"" << tableId_ << "\",\n"
            << "  \"table_hash\": \"" << engineGeometry.lookupOrDefault<word>("runtimeChemistryTableHash", "") << "\",\n"
            << "  \"runtime_mode\": \"" << engineGeometry.lookupOrDefault<word>("runtimeChemistryMode", "fullReducedKinetics") << "\",\n";
        if (missSummary.hasFirstTrustReject)
        {
            missFile
                << "  \"failure_class\": \"" << missSummary.failureClass << "\",\n"
                << "  \"failure_branch_id\": \"" << missSummary.failureBranchId << "\",\n"
                << "  \"reject_variable\": \"" << missSummary.rejectVariable << "\",\n"
                << "  \"reject_excess\": " << missSummary.rejectExcess << ",\n"
                << "  \"reject_nearest_sample_distance\": "
                << missSummary.rejectNearestSampleDistance << ",\n"
                << "  \"reject_state\": {\n";
            forAll(stateVariables_, dimi)
            {
                missFile
                    << "    \"" << stateVariables_[dimi] << "\": ";
                if (dimi < missSummary.rejectState.size())
                {
                    missFile << missSummary.rejectState[dimi];
                }
                else
                {
                    missFile << "0";
                }
                missFile << (dimi + 1 < stateVariables_.size() ? ",\n" : "\n");
            }
            missFile << "  },\n";
        }
        else
        {
            missFile
                << "  \"failure_class\": null,\n"
                << "  \"failure_branch_id\": null,\n"
                << "  \"reject_variable\": null,\n"
                << "  \"reject_excess\": null,\n"
                << "  \"reject_nearest_sample_distance\": null,\n"
                << "  \"reject_state\": {},\n";
        }
        missFile
            << "  \"first_offending_variables\": [";
        forAll(missSummary.offendingVariables, index)
        {
            if (index > 0)
            {
                missFile << ", ";
            }
            missFile << "\"" << missSummary.offendingVariables[index] << "\"";
        }
        missFile << "],\n";
        missFile << "  \"max_out_of_bound_by_variable\": {\n";
        forAll(missSummary.offendingVariables, index)
        {
            missFile
                << "    \"" << missSummary.offendingVariables[index] << "\": "
                << missSummary.maxOutOfBoundByVariable[index];
            missFile << (index + 1 < missSummary.offendingVariables.size() ? ",\n" : "\n");
        }
        missFile << "  },\n";
        missFile << "  \"miss_counts_by_variable\": {\n";
        forAll(missSummary.offendingVariables, index)
        {
            missFile
                << "    \"" << missSummary.offendingVariables[index] << "\": "
                << missSummary.offendingCounts[index];
            missFile << (index + 1 < missSummary.offendingVariables.size() ? ",\n" : "\n");
        }
        missFile << "  }\n"
                 << "}\n";
    }

    void noteFallbackTimestep() const
    {
        fallbackTimestepCount_ += 1;
    }

    void writeCoverageCorpusRowJson
    (
        OFstream& corpusFile,
        const CoverageCorpusRow& row,
        const bool useTrustRejectSnapshot
    ) const
    {
        const scalar denominator = max(row.queryCount, label(1));
        corpusFile << "    {\n";
        corpusFile << "      \"coverage_bucket_key\": [";
        forAll(row.coverageBucketKey, dimi)
        {
            if (dimi > 0)
            {
                corpusFile << ", ";
            }
            corpusFile << row.coverageBucketKey[dimi];
        }
        corpusFile << "],\n";
        if (useTrustRejectSnapshot)
        {
            corpusFile << "      \"high_fidelity_trust_reject\": true,\n";
        }
        corpusFile << "      \"raw_state\": {\n";
        forAll(stateVariables_, dimi)
        {
            const scalar rawValue =
                useTrustRejectSnapshot
              ? row.bestTrustRejectRawState[dimi]
              : row.rawStateSum[dimi]/denominator;
            corpusFile
                << "        \"" << stateVariables_[dimi] << "\": "
                << rawValue;
            corpusFile << (dimi + 1 < stateVariables_.size() ? ",\n" : "\n");
        }
        corpusFile << "      },\n";
        corpusFile << "      \"transformed_state\": {\n";
        forAll(stateVariables_, dimi)
        {
            const scalar transformedValue =
                useTrustRejectSnapshot
              ? row.bestTrustRejectTransformedState[dimi]
              : row.transformedStateSum[dimi]/denominator;
            corpusFile
                << "        \"" << stateVariables_[dimi] << "\": "
                << transformedValue;
            corpusFile << (dimi + 1 < stateVariables_.size() ? ",\n" : "\n");
        }
        corpusFile << "      },\n";
        corpusFile << "      \"query_count\": "
                   << (useTrustRejectSnapshot ? 1 : row.queryCount) << ",\n";
        corpusFile << "      \"table_hit_count\": "
                   << (useTrustRejectSnapshot ? 0 : row.tableHitCount) << ",\n";
        corpusFile << "      \"coverage_reject_count\": "
                   << (useTrustRejectSnapshot ? 0 : row.coverageRejectCount) << ",\n";
        corpusFile << "      \"trust_reject_count\": "
                   << (useTrustRejectSnapshot ? 1 : row.trustRejectCount) << ",\n";
        corpusFile << "      \"worst_reject_variable\": ";
        if (row.worstRejectVariable.size())
        {
            corpusFile << "\"" << row.worstRejectVariable << "\"";
        }
        else
        {
            corpusFile << "null";
        }
        corpusFile << ",\n";
        corpusFile << "      \"worst_reject_excess\": "
                   << (useTrustRejectSnapshot ? row.bestTrustRejectSnapshotExcess
                        : row.worstRejectExcess)
                   << ",\n";
        corpusFile << "      \"nearest_sample_distance_min\": "
                   << (row.hasNearestSampleDistance ? row.nearestSampleDistanceMin : 0.0)
                   << ",\n";
        corpusFile << "      \"nearest_sample_distance_max\": "
                   << (row.hasNearestSampleDistance ? row.nearestSampleDistanceMax : 0.0)
                   << ",\n";
        corpusFile << "      \"stage_names\": [";
        forAll(row.stageNames, stagei)
        {
            if (stagei > 0)
            {
                corpusFile << ", ";
            }
            corpusFile << "\"" << row.stageNames[stagei] << "\"";
        }
        corpusFile << "]\n";
        corpusFile << "    }";
    }

    void writeCoverageCorpus(const Time& runTime) const
    {
        if (!active_ || coverageRows_.empty())
        {
            return;
        }
        OFstream corpusFile(runTime.path()/"runtimeChemistryCoverageCorpus.json");
        corpusFile << "{\n";
        corpusFile << "  \"state_variables\": [";
        forAll(stateVariables_, dimi)
        {
            if (dimi > 0)
            {
                corpusFile << ", ";
            }
            corpusFile << "\"" << stateVariables_[dimi] << "\"";
        }
        corpusFile << "],\n";
        corpusFile << "  \"rows\": [\n";
        label emitted = 0;
        forAll(coverageRows_, rowi)
        {
            const CoverageCorpusRow& row = coverageRows_[rowi];
            if (emitted++)
            {
                corpusFile << ",\n";
            }
            writeCoverageCorpusRowJson(corpusFile, row, false);
            if (row.hasBestTrustRejectSnapshot)
            {
                if (emitted++)
                {
                    corpusFile << ",\n";
                }
                writeCoverageCorpusRowJson(corpusFile, row, true);
            }
        }
        corpusFile << "\n  ]\n";
        corpusFile << "}\n";
    }

    void writeSummary(const Time& runTime) const
    {
        if (!active_)
        {
            return;
        }
        writeCoverageCorpus(runTime);
        OFstream summaryFile
        (
            runTime.path()/("runtimeChemistrySummary." + runTime.timeName() + ".dat")
        );
        summaryFile
            << "# tableQueryCells tableHitCells fallbackTimesteps interpolationCacheHits coverageRejectCells trustRegionRejectCells\n"
            << tableQueryCells_ << tab
            << tableHitCells_ << tab
            << fallbackTimestepCount_ << tab
            << interpolationCacheHits_ << tab
            << coverageRejectCells_ << tab
            << trustRegionRejectCells_ << nl;
    }
};

} // End namespace Foam

int main(int argc, char *argv[])
{
    using namespace Foam;

    argList::addNote
    (
        "Transient chemistry-capable engine flow solver with spray cloud support,\n"
        "dynamic-mesh updates, and engine-style summary logging."
    );

    #include "postProcess.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"

    IOdictionary engineGeometry(readEngineGeometry(runTime));

    #include "createDynamicFvMesh.H"
    #include "createDyMControls.H"
    #include "createFields.H"
    #include "createFieldRefs.H"
    #include "createRhoUf.H"
    #include "compressibleCourantNo.H"
    #include "setInitialDeltaT.H"
    #include "initContinuityErrs.H"

    RuntimeChemistryTable runtimeChemistryTable(mesh, Y);

    PtrList<volScalarField> runtimeChemistrySources(Y.size());
    PtrList<volScalarField> runtimeChemistryDiagSources(Y.size());
    forAll(Y, speciesi)
    {
        runtimeChemistrySources.set
        (
            speciesi,
            new volScalarField
            (
                IOobject
                (
                    "runtimeChemistrySource_" + Y[speciesi].name(),
                    runTime.timeName(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                mesh,
                dimensionedScalar(dimMass/dimVolume/dimTime, Zero)
            )
        );
        runtimeChemistryDiagSources.set
        (
            speciesi,
            new volScalarField
            (
                IOobject
                (
                    "runtimeChemistryDiagSource_" + Y[speciesi].name(),
                    runTime.timeName(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                mesh,
                dimensionedScalar(dimMass/dimVolume/dimTime, Zero)
            )
        );
    }

    volScalarField runtimeChemistryQdot
    (
        IOobject
        (
            "runtimeChemistryQdot",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimEnergy/dimVolume/dimTime, Zero)
    );
    volScalarField runtimeChemistryQdotTemperatureSensitivity
    (
        IOobject
        (
            "runtimeChemistryQdotTemperatureSensitivity",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimEnergy/dimVolume/dimTime/dimTemperature, Zero)
    );

    turbulence->validate();

    Info<< "Initial cylinder mass: " << fvc::domainIntegrate(rho).value() << endl;
    Info<< "Initial crank angle: " << crankAngleDeg(engineGeometry, runTime) << " deg" << endl;
    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        if (engineGeometry.readIfModified())
        {
            Info<< "Reloaded engineGeometry\n" << endl;
        }

        #include "readDyMControls.H"

        {
            volScalarField divrhoU
            (
                "divrhoU",
                fvc::div(fvc::absolute(phi, rho, U))
            );

            #include "compressibleCourantNo.H"
            #include "setDeltaT.H"

            ++runTime;

            const scalar crankAngle = crankAngleDeg(engineGeometry, runTime);

            Info<< "Time = " << runTime.timeName() << " s" << nl
                << "Crank angle = " << crankAngle << " deg" << nl << endl;

            volVectorField rhoU("rhoU", rho*U);

            parcels.storeGlobalPositions();

            mesh.update();

            if (mesh.changing())
            {
                MRF.update();

                if (correctPhi)
                {
                    phi = mesh.Sf() & rhoUf;

                    #include "correctPhi.H"

                    fvc::makeRelative(phi, rho, U);
                }

                if (checkMeshCourantNo)
                {
                    #include "meshCourantNo.H"
                }
            }
        }

        parcels.evolve();

        #include "rhoEqn.H"

        while (pimple.loop())
        {
            #include "UEqn.H"
            tmp<fv::convectionScheme<scalar>> mvConvection
            (
                fv::convectionScheme<scalar>::New
                (
                    mesh,
                    fields,
                    phi,
                    mesh.divScheme("div(phi,Yi_h)")
                )
            );

            bool useRuntimeChemistryTable = false;
            if (runtimeChemistryTable.enabled(engineGeometry))
            {
                RuntimeChemistryTable::AuthorityMissSummary authorityMiss;
                useRuntimeChemistryTable = runtimeChemistryTable.populate
                (
                    runTime,
                    engineGeometry,
                    Y,
                    p,
                    thermo.T(),
                    runtimeChemistrySources,
                    runtimeChemistryDiagSources,
                    runtimeChemistryQdot,
                    runtimeChemistryQdotTemperatureSensitivity,
                    authorityMiss
                );
                if (useRuntimeChemistryTable)
                {
                    Qdot = runtimeChemistryQdot;
                    Info<< "Runtime chemistry table active: "
                        << runtimeChemistryTable.tableId() << endl;
                }
                else
                {
                    runtimeChemistryTable.writeAuthorityMiss(runTime, engineGeometry, authorityMiss);
                    if
                    (
                        runtimeChemistryTable.strictMode(engineGeometry)
                     && runtimeChemistryTable.abortOnAuthorityMiss(engineGeometry)
                    )
                    {
                        runtimeChemistryTable.writeSummary(runTime);
                        FatalErrorInFunction
                            << "Runtime chemistry table authority miss in strict mode for stage "
                            << engineGeometry.lookupOrDefault<word>("runtimeChemistryStageName", "")
                            << ". Uncovered cells=" << authorityMiss.uncoveredCells
                            << ", trust rejects=" << authorityMiss.trustRejectCells
                            << exit(FatalError);
                    }
                    runtimeChemistryTable.noteFallbackTimestep();
                    Info<< "Runtime chemistry table authority miss for "
                        << authorityMiss.uncoveredCells << " uncovered cells; max untracked mass fraction "
                        << authorityMiss.maxObservedUntrackedMassFraction
                        << "; trust-region rejects " << authorityMiss.trustRejectCells
                        << ". Falling back to "
                        << runtimeChemistryTable.fallbackPolicy() << endl;
                }
            }

            if (!useRuntimeChemistryTable)
            {
                combustion->correct();
                Qdot = combustion->Qdot();
            }

            volScalarField Yt(0.0*Y[0]);
            forAll(Y, i)
            {
                if (i != inertIndex && composition.active(i))
                {
                    volScalarField& Yi = Y[i];
                    if (useRuntimeChemistryTable)
                    {
                        const volScalarField runtimeSpeciesSu
                        (
                            "runtimeSpeciesSu",
                            runtimeChemistrySources[i] - runtimeChemistryDiagSources[i]*Yi
                        );
                        fvScalarMatrix YEqn
                        (
                            fvm::ddt(rho, Yi)
                          + mvConvection->fvmDiv(phi, Yi)
                          - fvm::laplacian(turbulence->muEff(), Yi)
                          - fvm::Sp(runtimeChemistryDiagSources[i], Yi)
                         ==
                            parcels.SYi(i, Yi)
                          + runtimeSpeciesSu
                          + fvOptions(rho, Yi)
                        );

                        YEqn.relax();

                        fvOptions.constrain(YEqn);

                        YEqn.solve("Yi");

                        fvOptions.correct(Yi);
                    }
                    else
                    {
                        fvScalarMatrix YEqn
                        (
                            fvm::ddt(rho, Yi)
                          + mvConvection->fvmDiv(phi, Yi)
                          - fvm::laplacian(turbulence->muEff(), Yi)
                         ==
                            parcels.SYi(i, Yi)
                          + combustion->R(Yi)
                          + fvOptions(rho, Yi)
                        );

                        YEqn.relax();

                        fvOptions.constrain(YEqn);

                        YEqn.solve("Yi");

                        fvOptions.correct(Yi);
                    }

                    Yi.clamp_min(0);
                    Yt += Yi;
                }
            }

            Y[inertIndex] = scalar(1) - Yt;
            Y[inertIndex].clamp_min(0);
            {
                volScalarField& he = thermo.he();
                tmp<volScalarField> tRuntimeQdotSp
                (
                    volScalarField::New
                    (
                        "runtimeQdotSp",
                        IOobject::NO_REGISTER,
                        mesh,
                        dimensionedScalar(dimMass/dimVolume/dimTime, Zero)
                    )
                );
                if (useRuntimeChemistryTable)
                {
                    const tmp<volScalarField> tCp = thermo.Cp();
                    volScalarField& runtimeQdotSp = tRuntimeQdotSp.ref();
                    scalarField& runtimeQdotSpCells = runtimeQdotSp.primitiveFieldRef();
                    const scalarField& qdotSensitivityCells =
                        runtimeChemistryQdotTemperatureSensitivity.primitiveField();
                    const scalarField& cpCells = tCp().primitiveField();
                    forAll(runtimeQdotSpCells, celli)
                    {
                        runtimeQdotSpCells[celli] = qdotSensitivityCells[celli]/max(cpCells[celli], scalar(1.0));
                    }
                    auto& runtimeQdotSpBoundary = runtimeQdotSp.boundaryFieldRef();
                    const auto& qdotSensitivityBoundary =
                        runtimeChemistryQdotTemperatureSensitivity.boundaryField();
                    const auto& cpBoundary = tCp().boundaryField();
                    forAll(runtimeQdotSpBoundary, patchi)
                    {
                        forAll(runtimeQdotSpBoundary[patchi], facei)
                        {
                            runtimeQdotSpBoundary[patchi][facei] =
                                qdotSensitivityBoundary[patchi][facei]
                               /max(cpBoundary[patchi][facei], scalar(1.0));
                        }
                    }
                }
                fvScalarMatrix EEqn
                (
                    useRuntimeChemistryTable
                  ? (
                        fvm::ddt(rho, he) + mvConvection->fvmDiv(phi, he)
                      + fvc::ddt(rho, K) + fvc::div(phi, K)
                      + (
                            he.name() == "e"
                          ? fvc::div
                            (
                                fvc::absolute(phi/fvc::interpolate(rho), U),
                                p,
                                "div(phiv,p)"
                            )
                          : -dpdt
                        )
                      - fvm::laplacian(turbulence->alphaEff(), he)
                      - fvm::Sp(tRuntimeQdotSp(), he)
                     ==
                        rho*(U&g)
                      + parcels.Sh(he)
                      + radiation->Sh(thermo, he)
                      + runtimeChemistryQdot - tRuntimeQdotSp()*he
                      + fvOptions(rho, he)
                    )
                  : (
                        fvm::ddt(rho, he) + mvConvection->fvmDiv(phi, he)
                      + fvc::ddt(rho, K) + fvc::div(phi, K)
                      + (
                            he.name() == "e"
                          ? fvc::div
                            (
                                fvc::absolute(phi/fvc::interpolate(rho), U),
                                p,
                                "div(phiv,p)"
                            )
                          : -dpdt
                        )
                      - fvm::laplacian(turbulence->alphaEff(), he)
                     ==
                        rho*(U&g)
                      + parcels.Sh(he)
                      + radiation->Sh(thermo, he)
                      + Qdot
                      + fvOptions(rho, he)
                    )
                );

                EEqn.relax();

                fvOptions.constrain(EEqn);

                EEqn.solve();

                fvOptions.correct(he);

                stabilizeThermoState(thermo, p, engineGeometry);
                thermo.correct();
                radiation->correct();

                Info<< "T gas min/max   " << min(T).value() << ", "
                    << max(T).value() << endl;
            }

            while (pimple.correct())
            {
                #include "pEqn.H"
            }

            stabilizePressureDensityState(p, rho, engineGeometry);

            if (pimple.turbCorr())
            {
                turbulence->correct();
            }
        }

        rho = thermo.rho();
        stabilizePressureDensityState(p, rho, engineGeometry);

        if (runTime.write())
        {
            combustion->Qdot()().write();
            runtimeChemistryTable.writeSummary(runTime);
            writeEngineSummary
            (
                runTime,
                crankAngleDeg(engineGeometry, runTime),
                p,
                thermo.T(),
                U,
                mesh
            );
        }

        runTime.printExecutionTime(Info);
    }

    Info<< "End\n" << endl;

    return 0;
}
