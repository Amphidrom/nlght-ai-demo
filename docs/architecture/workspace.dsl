workspace "nlght-ai Demo" "Versioned demo release architecture" {
    model {
        maintainer = person "Maintainer" "Creates a version tag"
        github = softwareSystem "GitHub" "Hosts source, Actions, and Releases" {
            workflow = container "Release workflow" "Builds and verifies wheel and sdist" "GitHub Actions"
            release = container "GitHub Release" "Stores versioned installable artifacts" "GitHub Releases"
        }
        user = person "Demo user" "Installs a selected demo version"

        maintainer -> workflow "Pushes v* tag"
        workflow -> release "Publishes wheel, sdist, and generated notes"
        user -> release "Downloads and installs a versioned wheel"
    }

    views {
        container github "ReleasePipeline" {
            include *
            autoLayout lr
        }
        styles {
            element "Person" { shape person }
            element "Container" { background #007a6a color #ffffff }
        }
    }
}
