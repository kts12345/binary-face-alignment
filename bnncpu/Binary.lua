local Binary, parent = torch.class('bnn.Binary','nn.Module')

function Binary:updateOutput(input)
    return input:sign()
end
