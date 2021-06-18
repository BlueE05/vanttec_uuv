#ifndef __UUV_ASMC_GUIDANCE_H__
#define __UUV_ASMC_GUIDANCE_H__

#include "asmc.hpp"

class ASMC_GUIDANCE// : public ASMC
{
    public:
        ASMC asmc;

        double Ka;
        double error_i;
        double desired_dot_error;
        double U;
        double Uax;

                // Constructor
        ASMC_GUIDANCE(double _sample_time_s, const double _Ka, const double _K2, const double _Kalpha, const double _Kmin, const double _miu, const DOFControllerType_E _type);

        // Destructor
        ~ASMC_GUIDANCE();

        void CalculateAuxiliaryControl(double _current);
};

#endif