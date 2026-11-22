/* Round trips through the installed library's C API; exits non-zero on any
 * difference from the expected bits. */
#include <omnicast/omnicast.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
    const double x[4] = { 1.0, 1.0 + 1.0 / 2048.0, 65504.0, 1.0e6 };
    /* FP16 nearest even: 1, 1 (halfway to 1 + 2^-10: the tie goes to even),
     * 65504 and +Inf */
    const uint16_t want[4] = { 0x3C00, 0x3C00, 0x7BFF, 0x7C00 };
    uint16_t h[4];
    double back[4];
    omc_options_t sat;

    omc_status_t s = omc_convert(OMC_FP64, x, OMC_FP16, h, 4, NULL);
    if( OMC_SUCCESS != s ) {
        printf("omc_convert: %s\n", omc_status_string(s));
        return 1;
    }
    if( 0 != memcmp(h, want, sizeof(want)) ) {
        printf("FP64 -> FP16: %04x %04x %04x %04x\n", h[0], h[1], h[2], h[3]);
        return 1;
    }
    memset(&sat, 0, sizeof(sat));
    sat.flags = OMC_FLAG_SATURATE;
    s = omc_convert(OMC_FP64, x, OMC_FP16, h, 4, &sat);
    if( OMC_SUCCESS != s || 0x7BFF != h[3] ) {
        printf("saturated FP64 -> FP16: %s, %04x\n", omc_status_string(s), h[3]);
        return 1;
    }
    s = omc_convert(OMC_FP16, h, OMC_FP64, back, 4, NULL);
    if( OMC_SUCCESS != s || 1.0 != back[0] || 65504.0 != back[3] ) {
        printf("FP16 -> FP64: %s, %g %g\n", omc_status_string(s), back[0], back[3]);
        return 1;
    }
    printf("OmniCast consumer: ok\n");
    return 0;
}
