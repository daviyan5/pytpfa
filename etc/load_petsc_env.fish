# =================================================================
# Usage: source load_petsc_env.fish --petsc_dir=<PATH> --mode=<OPT|DEBUG> --target=<TARGET>
# =================================================================

function load_petsc_main
    function show_usage
        echo "Error: Missing or invalid arguments."
        echo "Usage: source load_petsc_env.fish --petsc_dir=<PATH> --mode=<OPT|DEBUG> --target=<PERSONAL|APUANA|JARVIS>"
    end

    set PETSC_DIR_ARG ""
    set MODE_ARG ""
    set TARGET_ARG ""

    # 1. Parse Arguments
    for arg in $argv
        switch $arg
            case --petsc_dir=\*
                set PETSC_DIR_ARG (string split -m 1 = -- $arg)[2]
            case --mode=\*
                set MODE_ARG (string split -m 1 = -- $arg)[2]
            case --target=\*
                set TARGET_ARG (string split -m 1 = -- $arg)[2]
        end
    end

    if test -z "$PETSC_DIR_ARG"; or test -z "$MODE_ARG"; or test -z "$TARGET_ARG"
        show_usage
        return 1
    end

    # 2. Set Global Variables
    set -gx PETSC_DIR (realpath "$PETSC_DIR_ARG")
    set MODE (string upper -- "$MODE_ARG")
    set TARGET (string upper -- "$TARGET_ARG")
    set -g PETSC_ARCH ""

    # 3. Determine PETSC_ARCH
    if test "$MODE" = "DEBUG"
        if test "$TARGET" = "APUANA"
            set PETSC_ARCH "myconfiguredebugapuana"
        else if test "$TARGET" = "JARVIS"
            set PETSC_ARCH "myconfiguredebugjarvis"
        else
            set PETSC_ARCH "myconfiguredebug"
        end
    else
        if test "$TARGET" = "APUANA"
            set PETSC_ARCH "myconfigureoptapuana"
        else if test "$TARGET" = "JARVIS"
            set PETSC_ARCH "myconfigureoptjarvis"
        else
            set PETSC_ARCH "myconfigureopt"
        end
    end

    set -gx PETSC_ARCH $PETSC_ARCH

    # 4. Validation
    if not test -d "$PETSC_DIR"
        echo "Error: PETSc directory not found at '$PETSC_DIR'"
        return 1
    end

    if not test -d "$PETSC_DIR/$PETSC_ARCH"
        echo "Error: PETSC_ARCH directory '$PETSC_ARCH' not found inside PETSC_DIR."
        return 1
    end

    # =========================================================================
    # 5. CRITICAL: OpenMP and Threading Environment Variables
    #    PETSc was built with --with-openmp, so we MUST control thread count
    # =========================================================================
    set -gx OMP_NUM_THREADS 1
    set -gx OMP_PROC_BIND false
    set -gx OMP_PLACES threads
    set -gx OMP_STACKSIZE 64M
    
    # BLAS/LAPACK threading (Intel MKL, OpenBLAS, etc.)
    set -gx MKL_NUM_THREADS 1
    set -gx OPENBLAS_NUM_THREADS 1
    set -gx VECLIB_MAXIMUM_THREADS 1
    set -gx NUMEXPR_NUM_THREADS 1

    # 6. Update Paths
    echo "Loading Environment for: $PETSC_ARCH"

    # 6.1 Standard PETSc locations
    if not contains "$PETSC_DIR/$PETSC_ARCH/bin" $PATH
        set -gx PATH "$PETSC_DIR/$PETSC_ARCH/bin" $PATH
    end
    
    # Initialize LD_LIBRARY_PATH if empty
    if not set -q LD_LIBRARY_PATH
        set -gx LD_LIBRARY_PATH ""
    end

    if not string match -q "*$PETSC_DIR/$PETSC_ARCH/lib*" -- $LD_LIBRARY_PATH
        set -gx LD_LIBRARY_PATH "$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH"
    end

    # 6.2 Auto-Link External Packages (e.g., mpich, hdf5)
    set EXT_PACKAGES_DIR "$PETSC_DIR/$PETSC_ARCH/externalpackages"
    
    if test -d "$EXT_PACKAGES_DIR"
        echo "  > Scanning external packages in: $EXT_PACKAGES_DIR"
        
        for pkg in $EXT_PACKAGES_DIR/*
            if test -d "$pkg"
                if test -d "$pkg/bin"
                    if not contains "$pkg/bin" $PATH
                        set -gx PATH "$pkg/bin" $PATH
                        echo "    + Added binary path: "(basename $pkg)"/bin"
                    end
                end
                
                if test -d "$pkg/lib"
                    if not string match -q "*$pkg/lib*" -- $LD_LIBRARY_PATH
                        set -gx LD_LIBRARY_PATH "$pkg/lib:$LD_LIBRARY_PATH"
                        echo "    + Added library path: "(basename $pkg)"/lib"
                    end
                end
            end
        end
    end

    # 6.3 PKG_CONFIG_PATH
    if set -q PKG_CONFIG_PATH
        if not string match -q "*$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig*" -- $PKG_CONFIG_PATH
            set -gx PKG_CONFIG_PATH "$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig:$PKG_CONFIG_PATH"
        end
    else
        set -gx PKG_CONFIG_PATH "$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig"
    end

    # 7. Handle petsc4py
    if set -q PYTHONPATH
        if not string match -q "*$PETSC_DIR/$PETSC_ARCH/lib*" -- $PYTHONPATH
            set -gx PYTHONPATH "$PETSC_DIR/$PETSC_ARCH/lib:$PYTHONPATH"
        end
    else
        set -gx PYTHONPATH "$PETSC_DIR/$PETSC_ARCH/lib"
    end

    # 8. Verification
    echo "--------------------------------------------------------"
    echo "Environment Updated (Fish Shell)"
    echo "--------------------------------------------------------"
    echo "  PETSC_DIR:        $PETSC_DIR"
    echo "  PETSC_ARCH:       $PETSC_ARCH"
    echo "  OMP_NUM_THREADS:  $OMP_NUM_THREADS"

    if command -v mpiexec > /dev/null
        echo "  ✔ MPI executable found: "(command -v mpiexec)
    else
        echo "  ⚠ Warning: 'mpiexec' not found in PATH."
    end

    if python3 -c "import petsc4py" 2>/dev/null
        echo "  ✔ Python import 'petsc4py' successful."
    else
        echo "  ⚠ Warning: Python could not import petsc4py."
    end
end

# Execute main function
load_petsc_main $argv